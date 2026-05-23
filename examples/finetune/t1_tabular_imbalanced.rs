//! # Tier 1.4 — Tabular classification — imbalanced
//!
//! Falsifier: 2-class macro-F1 ≥ 0.3 on well-separated synthetic.
//!
//! Run with: cargo run --example t1_tabular_imbalanced

use apr_cookbook::finetune::tabular_classification as tc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_imbalanced/data.jsonl";
const N_CLASSES: u32 = 2;
const N_FEATURES: usize = 2;
const F1_FLOOR: f64 = 0.3;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_imbalanced")?;
    let samples = tc::load_samples(FIXTURE, N_FEATURES)?;
    let means = tc::fit_ncm(&samples, N_CLASSES);
    let preds: Vec<u32> = samples
        .iter()
        .map(|s| tc::predict_ncm(&means, &s.features))
        .collect();
    let labels: Vec<u32> = samples.iter().map(|s| s.label).collect();
    let f1 = tc::macro_f1(&preds, &labels, N_CLASSES);
    println!("✓ imbalanced NCM: macro-F1={f1:.4} (floor {F1_FLOOR})");
    assert!(f1 >= F1_FLOOR, "falsifier: F1 {f1} should be ≥ {F1_FLOOR}");
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
        let s = tc::load_samples(FIXTURE, N_FEATURES).expect("load");
        let m = tc::fit_ncm(&s, N_CLASSES);
        let p: Vec<u32> = s.iter().map(|x| tc::predict_ncm(&m, &x.features)).collect();
        let l: Vec<u32> = s.iter().map(|x| x.label).collect();
        assert!(tc::macro_f1(&p, &l, N_CLASSES) >= F1_FLOOR);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Predict always class 0 — minority classes get F1=0; macro-F1 drops.
        let s = tc::load_samples(FIXTURE, N_FEATURES).expect("load");
        let p = vec![0u32; s.len()];
        let l: Vec<u32> = s.iter().map(|x| x.label).collect();
        assert!(tc::macro_f1(&p, &l, N_CLASSES) < F1_FLOOR);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tc::load_samples(FIXTURE, N_FEATURES).expect("a");
        let b = tc::load_samples(FIXTURE, N_FEATURES).expect("b");
        assert_eq!(tc::fit_ncm(&a, N_CLASSES), tc::fit_ncm(&b, N_CLASSES));
    }
}
