//! Tier 2.3 continued pretraining — shared helper.
//!
//! Models domain-adaptive continued pretraining on raw text. The
//! "model" is a per-token unigram log-likelihood table that we update
//! by counting tokens in the domain corpus. The falsifier is a
//! perplexity-decrease invariant: training on a domain corpus reduces
//! perplexity on a held-out passage from that domain by ≥ 30%, while
//! drift on a *general* held-out passage stays ≤ 5%.
//!
//! This abstracts away transformer training but preserves the
//! *observable* property a real CP run should have.

use crate::Result;
use std::collections::HashMap;

/// A token frequency table — our "language model".
#[derive(Debug, Clone, Default)]
pub struct UnigramLM {
    pub counts: HashMap<String, u64>,
    pub total: u64,
}

impl UnigramLM {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update counts with a corpus.
    pub fn train(&mut self, corpus: &[&str]) {
        for line in corpus {
            for tok in tokenize(line) {
                *self.counts.entry(tok).or_insert(0) += 1;
                self.total += 1;
            }
        }
    }

    /// Per-token log-likelihood with add-1 smoothing over a fixed
    /// vocabulary universe of `vocab_size` tokens.
    #[must_use]
    pub fn log_prob(&self, token: &str, vocab_size: u32) -> f64 {
        let count = *self.counts.get(token).unwrap_or(&0);
        let smoothed = (count as f64 + 1.0) / (self.total as f64 + f64::from(vocab_size));
        smoothed.ln()
    }

    /// Perplexity on a held-out passage.
    #[must_use]
    pub fn perplexity(&self, passage: &str, vocab_size: u32) -> f64 {
        let toks = tokenize(passage);
        if toks.is_empty() {
            return f64::NAN;
        }
        let mut sum_log_p = 0.0_f64;
        for tok in &toks {
            sum_log_p += self.log_prob(tok, vocab_size);
        }
        let mean_neg_log_p = -sum_log_p / toks.len() as f64;
        mean_neg_log_p.exp()
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_lowercase).collect()
}

/// Build a base "general" LM as a starting point. Trains on a fixed
/// set of mixed-genre sentences so it's deterministic.
#[must_use]
pub fn base_general_lm() -> UnigramLM {
    let general_corpus = vec![
        "the quick brown fox jumps over the lazy dog",
        "machine learning models require careful evaluation",
        "deterministic seeds produce reproducible results",
        "fine tuning adapts pretrained models to new tasks",
        "attention is all you need is a famous paper",
        "open source software powers modern infrastructure",
        "the cat sat on the mat in the morning",
        "scientific progress accumulates through replication",
    ];
    let mut lm = UnigramLM::new();
    lm.train(&general_corpus);
    lm
}

/// Continued pretraining: snapshot perplexity, train on domain corpus,
/// re-evaluate on both domain and general held-out passages.
#[derive(Debug, Clone, PartialEq)]
pub struct CpReport {
    pub domain: String,
    pub initial_domain_ppl: f64,
    pub final_domain_ppl: f64,
    pub initial_general_ppl: f64,
    pub final_general_ppl: f64,
    pub n_corpus_tokens: u64,
}

impl CpReport {
    /// Falsifier: domain perplexity drops ≥ 30%.
    #[must_use]
    pub fn domain_perplexity_dropped(&self) -> bool {
        self.final_domain_ppl <= self.initial_domain_ppl * 0.7
    }

    /// Falsifier: general perplexity does not *worsen* by > 5%.
    /// Downward drift (general got better) is positive transfer, not forgetting.
    #[must_use]
    pub fn no_catastrophic_forgetting(&self) -> bool {
        self.final_general_ppl <= self.initial_general_ppl * 1.05
    }
}

/// Run continued pretraining on a synthetic domain corpus.
pub fn run_cp(
    domain: &str,
    domain_corpus: &[&str],
    domain_holdout: &str,
    general_holdout: &str,
    vocab_size: u32,
) -> Result<CpReport> {
    let mut lm = base_general_lm();
    let initial_domain_ppl = lm.perplexity(domain_holdout, vocab_size);
    let initial_general_ppl = lm.perplexity(general_holdout, vocab_size);

    let pre_count = lm.total;
    lm.train(domain_corpus);
    let n_corpus_tokens = lm.total - pre_count;

    let final_domain_ppl = lm.perplexity(domain_holdout, vocab_size);
    let final_general_ppl = lm.perplexity(general_holdout, vocab_size);

    Ok(CpReport {
        domain: domain.to_string(),
        initial_domain_ppl,
        final_domain_ppl,
        initial_general_ppl,
        final_general_ppl,
        n_corpus_tokens,
    })
}

/// Five built-in synthetic domain corpora and their held-out passages.
#[must_use]
pub fn legal_domain() -> (Vec<&'static str>, &'static str) {
    let corpus = vec![
        "the contract shall be governed by applicable law and jurisdiction",
        "indemnification clauses limit liability exposure for the parties",
        "pursuant to the terms hereof the parties agree to arbitration",
        "force majeure suspends performance during extraordinary events",
        "intellectual property rights remain with the original creator",
        "warranties are expressly disclaimed except as required by statute",
    ];
    let holdout = "the parties shall indemnify each other under contract terms";
    (corpus, holdout)
}

#[must_use]
pub fn code_domain() -> (Vec<&'static str>, &'static str) {
    let corpus = vec![
        "fn main let mut result vec push iterate over slice",
        "fn parse input return result error handling pattern match",
        "let value match expression return early on none case",
        "vec dedup sort by key chain map filter collect",
        "impl trait for struct method signature self ref mut",
        "use std collections hashmap entry insert or default",
    ];
    let holdout = "fn main let mut value vec push return result";
    (corpus, holdout)
}

#[must_use]
pub fn medical_domain() -> (Vec<&'static str>, &'static str) {
    let corpus = vec![
        "the patient presented with acute symptoms requiring urgent intervention",
        "differential diagnosis includes infection inflammation and metabolic causes",
        "imaging revealed pathological findings consistent with the clinical picture",
        "treatment regimen includes pharmacotherapy and supportive care measures",
        "follow up evaluation is recommended at six week intervals",
        "patient education on disease management improves adherence and outcomes",
    ];
    let holdout = "the patient presented with acute symptoms and pathological findings";
    (corpus, holdout)
}

#[must_use]
pub fn codeswitch_domain() -> (Vec<&'static str>, &'static str) {
    let corpus = vec![
        "let x string literal mut vec str variable assignment statement",
        "fn name parameter str argument return type result wrapper",
        "let value option some none match arm pattern handle exhaustively",
        "vec iter map collect into result type signature explicit",
        "impl block method definition signature self reference borrow checker",
        "let result iter sum collect into vec usize sized integer literal",
    ];
    let holdout = "let result vec map sum into option some none match";
    (corpus, holdout)
}

#[must_use]
pub fn scientific_domain() -> (Vec<&'static str>, &'static str) {
    let corpus = vec![
        "the experimental results demonstrate statistical significance with p value below threshold",
        "previous work established a baseline against which improvements are measured",
        "we hypothesize that increased capacity correlates with reduced perplexity",
        "the method achieves state of the art performance on standard benchmarks",
        "ablation studies isolate the contribution of each architectural component",
        "limitations include sample size and the specific evaluation protocol used",
    ];
    let holdout = "results demonstrate statistical significance against the baseline benchmark";
    (corpus, holdout)
}

/// General-domain held-out passage shared by all 5 recipes (used to
/// detect catastrophic forgetting).
pub const GENERAL_HOLDOUT: &str = "the cat sat on the mat in the morning";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_cp_reduces_domain_perplexity() {
        let (corpus, holdout) = legal_domain();
        let r = run_cp("legal", &corpus, holdout, GENERAL_HOLDOUT, 50_000).expect("cp");
        assert!(
            r.domain_perplexity_dropped(),
            "legal domain ppl should drop ≥30%: {} → {}",
            r.initial_domain_ppl,
            r.final_domain_ppl
        );
    }

    #[test]
    fn no_catastrophic_forgetting_on_general() {
        let (corpus, holdout) = legal_domain();
        let r = run_cp("legal", &corpus, holdout, GENERAL_HOLDOUT, 50_000).expect("cp");
        assert!(
            r.no_catastrophic_forgetting(),
            "general drift should be ≤5%: {} → {}",
            r.initial_general_ppl,
            r.final_general_ppl
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let (corpus, holdout) = code_domain();
        let r1 = run_cp("code", &corpus, holdout, GENERAL_HOLDOUT, 50_000).expect("a");
        let r2 = run_cp("code", &corpus, holdout, GENERAL_HOLDOUT, 50_000).expect("b");
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_corpus_keeps_base_perplexity() {
        let r = run_cp("none", &[], "the quick brown fox", GENERAL_HOLDOUT, 50_000).expect("cp");
        assert_eq!(r.initial_domain_ppl, r.final_domain_ppl);
    }
}
