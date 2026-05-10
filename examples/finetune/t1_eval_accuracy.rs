//! # Tier 1.2 — Eval primitive — Accuracy
//!
//! Falsifier: accuracy on perfectly-predicted holdout = 1.0.
//!
//! Demonstrates the **t1_eval_accuracy** recipe per
//! `docs/specifications/fine-tuning-cookbook.md` (PMAT-332).
//!
//! Run with: cargo run --example t1_eval_accuracy

use apr_cookbook::finetune::eval_primitives;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_eval_accuracy/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_eval_accuracy")?;
    let (preds, labels) = eval_primitives::load_int_pairs(FIXTURE)?;
    let acc = eval_primitives::accuracy(&preds, &labels);
    println!(
        "✓ accuracy on {n} samples = {acc} (perfect predictions)",
        n = preds.len()
    );
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "falsifier broke: accuracy on perfect-pred fixture should be 1.0, got {acc}"
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
        let (p, l) = eval_primitives::load_int_pairs(FIXTURE).expect("load");
        let acc = eval_primitives::accuracy(&p, &l);
        assert!((acc - 1.0).abs() < 1e-12, "got {acc}");
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Flip every prediction — accuracy should drop to near 0 (not 1.0).
        let (p, l) = eval_primitives::load_int_pairs(FIXTURE).expect("load");
        let perturbed: Vec<u32> = p.iter().map(|&v| (v + 1) % 5).collect();
        let acc = eval_primitives::accuracy(&perturbed, &l);
        assert!(
            acc < 1.0,
            "perturbed should drop accuracy below 1.0, got {acc}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = eval_primitives::load_int_pairs(FIXTURE).expect("a");
        let b = eval_primitives::load_int_pairs(FIXTURE).expect("b");
        assert_eq!(a, b);
    }
}
