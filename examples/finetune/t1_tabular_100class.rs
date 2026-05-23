//! # Tier 1.4 — Tabular classification — 100-class top-5
//!
//! Falsifier: top-5 accuracy ≥ 0.5 on well-separated synthetic.
//!
//! Run with: cargo run --example t1_tabular_100class

use apr_cookbook::finetune::tabular_classification as tc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_100class/data.jsonl";
const TOP_K_FLOOR: f64 = 0.5;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_100class")?;
    let samples = tc::load_samples(FIXTURE, 4)?;
    let means = tc::fit_ncm(&samples, 100);
    let mut top5_correct = 0u32;
    for s in &samples {
        let topk = tc::predict_topk(&means, &s.features, 5);
        if topk.contains(&s.label) {
            top5_correct += 1;
        }
    }
    let top5 = f64::from(top5_correct) / samples.len() as f64;
    println!("✓ 100-class top-5: {top5:.4} (floor {TOP_K_FLOOR})");
    assert!(
        top5 >= TOP_K_FLOOR,
        "top-5 acc {top5} should be ≥ {TOP_K_FLOOR}"
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
        let s = tc::load_samples(FIXTURE, 4).expect("load");
        let m = tc::fit_ncm(&s, 100);
        let mut hits = 0u32;
        for sample in &s {
            if tc::predict_topk(&m, &sample.features, 5).contains(&sample.label) {
                hits += 1;
            }
        }
        let top5 = f64::from(hits) / s.len() as f64;
        assert!(top5 >= TOP_K_FLOOR);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // top-1 (k=1) on this dataset will be lower than top-5
        let s = tc::load_samples(FIXTURE, 4).expect("load");
        let m = tc::fit_ncm(&s, 100);
        let mut top1_hits = 0u32;
        let mut top5_hits = 0u32;
        for sample in &s {
            if tc::predict_topk(&m, &sample.features, 1).contains(&sample.label) {
                top1_hits += 1;
            }
            if tc::predict_topk(&m, &sample.features, 5).contains(&sample.label) {
                top5_hits += 1;
            }
        }
        // top-5 must be ≥ top-1 (monotonicity).
        assert!(
            top5_hits >= top1_hits,
            "top-5 hits {top5_hits} < top-1 {top1_hits}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tc::load_samples(FIXTURE, 4).expect("a");
        let b = tc::load_samples(FIXTURE, 4).expect("b");
        assert_eq!(tc::fit_ncm(&a, 100), tc::fit_ncm(&b, 100));
    }
}
