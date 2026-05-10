//! # Tier 1.2 — Eval primitive — ROUGE-L
//!
//! Falsifier: ROUGE-L of identical strings = 1.0; ROUGE-L of disjoint
//! strings = 0.0.
//!
//! Run with: cargo run --example t1_eval_rouge_l

use apr_cookbook::finetune::eval_primitives;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_eval_rouge_l/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_eval_rouge_l")?;
    let pairs = eval_primitives::load_string_pairs(FIXTURE)?;

    let mut identical_count = 0;
    let mut disjoint_count = 0;
    for (i, (r, h)) in pairs.iter().enumerate() {
        let score = eval_primitives::rouge_l(r, h);
        if r == h {
            assert!((score - 1.0).abs() < 1e-12, "identical pair {i}: {score}");
            identical_count += 1;
        } else {
            assert!(score.abs() < 1e-12, "disjoint pair {i}: {score}");
            disjoint_count += 1;
        }
    }
    println!(
        "✓ ROUGE-L: {identical_count} identical pairs scored 1.0, \
         {disjoint_count} disjoint pairs scored 0.0"
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
            let s = eval_primitives::rouge_l(r, h);
            if r == h {
                assert!((s - 1.0).abs() < 1e-12);
            } else {
                assert!(s.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Half-overlap: ROUGE-L between [a, b, c, d] and [a, b, x, y] is 0.5
        let r: Vec<String> = "a b c d".split_whitespace().map(String::from).collect();
        let h: Vec<String> = "a b x y".split_whitespace().map(String::from).collect();
        let s = eval_primitives::rouge_l(&r, &h);
        assert!(
            s > 0.0 && s < 1.0,
            "half-overlap should be in (0,1), got {s}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let pairs = eval_primitives::load_string_pairs(FIXTURE).expect("load");
        let (r, h) = &pairs[0];
        assert_eq!(
            eval_primitives::rouge_l(r, h),
            eval_primitives::rouge_l(r, h)
        );
    }
}
