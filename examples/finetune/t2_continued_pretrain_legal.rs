//! # Tier 2.3 — Continued pretraining — Legal corpus (llama family)
//!
//! Falsifier: domain perplexity on legal-corpus-mini drops ≥ 30% after CP;
//! general drift on a held-out general passage stays ≤ 5%.
//!
//! Run with: cargo run --example t2_continued_pretrain_legal

use apr_cookbook::finetune::continued_pretrain as cp;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const VOCAB_SIZE: u32 = 50_000;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_continued_pretrain_legal")?;
    let (corpus, holdout) = cp::legal_domain();
    let report = cp::run_cp("legal", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE)?;
    println!(
        "✓ legal CP: domain ppl {:.1} → {:.1} (general {:.1} → {:.1}, {} domain tokens)",
        report.initial_domain_ppl,
        report.final_domain_ppl,
        report.initial_general_ppl,
        report.final_general_ppl,
        report.n_corpus_tokens
    );
    assert!(
        report.domain_perplexity_dropped(),
        "falsifier: legal domain ppl must drop ≥ 30%"
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
        let (corpus, holdout) = cp::legal_domain();
        let r = cp::run_cp("legal", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert!(r.domain_perplexity_dropped() && r.no_catastrophic_forgetting());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty corpus: no learning happens, so domain ppl does NOT drop.
        let (_, holdout) = cp::legal_domain();
        let r = cp::run_cp("legal", &[], holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert!(
            !r.domain_perplexity_dropped(),
            "empty corpus must not satisfy ≥30% ppl drop"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let (corpus, holdout) = cp::legal_domain();
        let a = cp::run_cp("legal", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        let b = cp::run_cp("legal", &corpus, holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert_eq!(a, b);
    }
}
