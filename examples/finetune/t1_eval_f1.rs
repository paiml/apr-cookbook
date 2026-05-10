//! # Tier 1.2 — Eval primitive — F1
//!
//! Falsifier: F1 on balanced perfect predictions = 1.0; on always-majority
//! predictor < 1.0.
//!
//! Run with: cargo run --example t1_eval_f1

use apr_cookbook::finetune::eval_primitives;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_eval_f1/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_eval_f1")?;
    let (preds, labels) = eval_primitives::load_int_pairs(FIXTURE)?;
    let f1 = eval_primitives::f1_binary(&preds, &labels);
    let always_one: Vec<u32> = vec![1; labels.len()];
    let f1_majority = eval_primitives::f1_binary(&always_one, &labels);
    println!("✓ F1 perfect={f1}, always-majority={f1_majority}");
    assert!(
        (f1 - 1.0).abs() < 1e-12,
        "falsifier 1: F1 on perfect should be 1.0, got {f1}"
    );
    assert!(
        f1_majority < 1.0,
        "falsifier 2: always-majority predictor should yield F1 < 1.0, got {f1_majority}"
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
        let f1 = eval_primitives::f1_binary(&p, &l);
        assert!((f1 - 1.0).abs() < 1e-12);
        let always_one: Vec<u32> = vec![1; l.len()];
        let f1m = eval_primitives::f1_binary(&always_one, &l);
        assert!(f1m < 1.0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Inverting predictions on balanced data yields F1 ≈ 0.
        let (p, l) = eval_primitives::load_int_pairs(FIXTURE).expect("load");
        let inverted: Vec<u32> = p.iter().map(|&v| 1 - v).collect();
        let f1 = eval_primitives::f1_binary(&inverted, &l);
        assert!(f1 < 0.5, "inverted predictions should crater F1, got {f1}");
    }

    #[test]
    fn deterministic_across_runs() {
        let (p, l) = eval_primitives::load_int_pairs(FIXTURE).expect("load");
        assert_eq!(
            eval_primitives::f1_binary(&p, &l),
            eval_primitives::f1_binary(&p, &l)
        );
    }
}
