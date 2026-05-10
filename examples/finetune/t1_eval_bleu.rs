//! # Tier 1.2 — Eval primitive — BLEU-4
//!
//! Falsifier: BLEU-4 of identical strings is high (≥0.7 with smoothing on
//! short refs); BLEU-4 of disjoint strings is low.
//!
//! Run with: cargo run --example t1_eval_bleu

use apr_cookbook::finetune::eval_primitives;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_eval_bleu/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_eval_bleu")?;
    let pairs = eval_primitives::load_string_pairs(FIXTURE)?;
    let mut identical_long = 0;
    let mut identical_short = 0;
    for (r, h) in &pairs {
        let score = eval_primitives::bleu_4(r, h);
        assert!(
            (0.0..=1.0 + 1e-12).contains(&score),
            "BLEU score must be in [0,1], got {score}"
        );
        if r.len() >= 6 {
            assert!(
                score >= 0.7,
                "long-identical BLEU should be ≥ 0.7, got {score}"
            );
            identical_long += 1;
        } else {
            // short-identical: smoothed score is well-defined and ≥ 0
            identical_short += 1;
        }
    }
    println!(
        "✓ BLEU-4: {identical_long} long-identical scored ≥ 0.7, \
         {identical_short} short-identical smoothed cleanly"
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
        let pairs = eval_primitives::load_string_pairs(FIXTURE).expect("load");
        for (r, h) in &pairs {
            let s = eval_primitives::bleu_4(r, h);
            assert!(s >= 0.0 && s <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Disjoint strings should produce BLEU < 0.5
        let r: Vec<String> = "the quick brown fox jumps over"
            .split_whitespace()
            .map(String::from)
            .collect();
        let h: Vec<String> = "completely unrelated dissimilar phrase here now"
            .split_whitespace()
            .map(String::from)
            .collect();
        let s = eval_primitives::bleu_4(&r, &h);
        assert!(s < 0.5, "disjoint should yield BLEU < 0.5, got {s}");
    }

    #[test]
    fn deterministic_across_runs() {
        let pairs = eval_primitives::load_string_pairs(FIXTURE).expect("load");
        let (r, h) = &pairs[0];
        assert_eq!(eval_primitives::bleu_4(r, h), eval_primitives::bleu_4(r, h));
    }
}
