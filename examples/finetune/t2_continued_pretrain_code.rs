//! # Tier 2.3 — Continued pretraining — Code corpus (mistral family)
//!
//! Falsifier: code perplexity drops on code-corpus-mini; non-code perplexity
//! drift ≤ 5% (positive transfer counts as well-behaved, not forgetting).
//!
//! Run with: cargo run --example t2_continued_pretrain_code

use apr_cookbook::finetune::continued_pretrain as cp;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const VOCAB_SIZE: u32 = 50_000;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_continued_pretrain_code")?;
    let (corpus, holdout) = cp::code_domain();
    let report = cp::run_cp("code", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE)?;
    println!(
        "✓ code CP: domain ppl {:.1} → {:.1} (general {:.1} → {:.1}, {} domain tokens)",
        report.initial_domain_ppl,
        report.final_domain_ppl,
        report.initial_general_ppl,
        report.final_general_ppl,
        report.n_corpus_tokens
    );
    assert!(
        report.domain_perplexity_dropped(),
        "falsifier: code domain ppl must drop ≥ 30%"
    );
    assert!(
        report.no_catastrophic_forgetting(),
        "falsifier: general ppl must not worsen > 5%"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let (corpus, holdout) = cp::code_domain();
        let r = cp::run_cp("code", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert!(r.domain_perplexity_dropped() && r.no_catastrophic_forgetting());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let (_, holdout) = cp::code_domain();
        let r = cp::run_cp("code", &[], holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert!(!r.domain_perplexity_dropped());
    }

    #[test]
    fn deterministic_across_runs() {
        let (corpus, holdout) = cp::code_domain();
        let a = cp::run_cp("code", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        let b = cp::run_cp("code", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert_eq!(a, b);
    }
}
