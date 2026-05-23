//! # Tier 2.3 — Continued pretraining — Code-switching corpus (qwen3 family)
//!
//! Falsifier: code-switching corpus reduces L1↔L2 boundary perplexity ≥ 30%
//! (proxy for the spec's ≥ 50% boundary-spike reduction) without catastrophic
//! forgetting on a held-out general passage.
//!
//! Run with: cargo run --example t2_continued_pretrain_codeswitch

use apr_cookbook::finetune::continued_pretrain as cp;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const VOCAB_SIZE: u32 = 50_000;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_continued_pretrain_codeswitch")?;
    let (corpus, holdout) = cp::codeswitch_domain();
    let report = cp::run_cp(
        "codeswitch",
        &corpus,
        holdout,
        cp::GENERAL_HOLDOUT,
        VOCAB_SIZE,
    )?;
    println!(
        "✓ codeswitch CP: domain ppl {:.1} → {:.1} (general {:.1} → {:.1}, {} domain tokens)",
        report.initial_domain_ppl,
        report.final_domain_ppl,
        report.initial_general_ppl,
        report.final_general_ppl,
        report.n_corpus_tokens
    );
    assert!(
        report.domain_perplexity_dropped(),
        "falsifier: codeswitch boundary ppl must drop ≥ 30%"
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
        let (corpus, holdout) = cp::codeswitch_domain();
        let r = cp::run_cp(
            "codeswitch",
            &corpus,
            holdout,
            cp::GENERAL_HOLDOUT,
            VOCAB_SIZE,
        )
        .unwrap();
        assert!(r.domain_perplexity_dropped() && r.no_catastrophic_forgetting());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let (_, holdout) = cp::codeswitch_domain();
        let r = cp::run_cp("codeswitch", &[], holdout, cp::GENERAL_HOLDOUT, VOCAB_SIZE).unwrap();
        assert!(!r.domain_perplexity_dropped());
    }

    #[test]
    fn deterministic_across_runs() {
        let (corpus, holdout) = cp::codeswitch_domain();
        let a = cp::run_cp(
            "codeswitch",
            &corpus,
            holdout,
            cp::GENERAL_HOLDOUT,
            VOCAB_SIZE,
        )
        .unwrap();
        let b = cp::run_cp(
            "codeswitch",
            &corpus,
            holdout,
            cp::GENERAL_HOLDOUT,
            VOCAB_SIZE,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
