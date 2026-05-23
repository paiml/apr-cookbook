//! # Tier 1.4 — Tabular classification — Binary (linearly separable)
//!
//! Falsifier: binary classifier accuracy on linearly-separable synthetic = 1.0.
//!
//! Run with: cargo run --example t1_tabular_binary

use apr_cookbook::finetune::tabular_classification as tc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_binary/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_binary")?;
    let samples = tc::load_samples(FIXTURE, 2)?;
    let means = tc::fit_ncm(&samples, 2);
    let preds: Vec<u32> = samples
        .iter()
        .map(|s| tc::predict_ncm(&means, &s.features))
        .collect();
    let labels: Vec<u32> = samples.iter().map(|s| s.label).collect();
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count();
    let acc = correct as f64 / preds.len() as f64;
    println!("✓ binary NCM: {correct}/{} correct = {acc:.4}", preds.len());
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "falsifier: separable should be 1.0, got {acc}"
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
        let s = tc::load_samples(FIXTURE, 2).expect("load");
        let m = tc::fit_ncm(&s, 2);
        let p: Vec<u32> = s.iter().map(|x| tc::predict_ncm(&m, &x.features)).collect();
        let l: Vec<u32> = s.iter().map(|x| x.label).collect();
        let correct = p.iter().zip(l.iter()).filter(|(a, b)| a == b).count();
        assert_eq!(correct, s.len());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Swap labels on the fixture: NCM means swap, and prediction-vs-label flips.
        let mut s = tc::load_samples(FIXTURE, 2).expect("load");
        for x in &mut s {
            x.label = 1 - x.label;
        }
        let m = tc::fit_ncm(&s, 2);
        let p: Vec<u32> = s.iter().map(|x| tc::predict_ncm(&m, &x.features)).collect();
        let labels_after_swap: Vec<u32> = s.iter().map(|x| x.label).collect();
        let correct = p
            .iter()
            .zip(labels_after_swap.iter())
            .filter(|(a, b)| a == b)
            .count();
        // After re-fit on swapped data, NCM still achieves perfect accuracy (NCM is symmetric).
        // The perturbation we want to test: inject 10 mislabeled points.
        let mut sabotaged = s.clone();
        for x in sabotaged.iter_mut().take(10) {
            x.label = 1 - x.label;
        }
        let m2 = tc::fit_ncm(&sabotaged, 2);
        let p2: Vec<u32> = sabotaged
            .iter()
            .map(|x| tc::predict_ncm(&m2, &x.features))
            .collect();
        let l2: Vec<u32> = sabotaged.iter().map(|x| x.label).collect();
        let acc2 =
            p2.iter().zip(l2.iter()).filter(|(a, b)| a == b).count() as f64 / p2.len() as f64;
        assert!(
            acc2 < 1.0,
            "sabotaged should drop accuracy below 1.0, got {acc2}"
        );
        let _ = correct;
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tc::load_samples(FIXTURE, 2).expect("a");
        let b = tc::load_samples(FIXTURE, 2).expect("b");
        assert_eq!(tc::fit_ncm(&a, 2), tc::fit_ncm(&b, 2));
    }
}
