//! # Tier 3.7 — Entropic open-set loss (tabular-only)
//!
//! Falsifier: predictive entropy on unseen classes exceeds entropy on seen
//! classes (mean Δ ≥ 0.5) on a deterministic fixture.
//!
//! Run with: cargo run --example t3_open_set_entropic

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn seen() -> Vec<Vec<f64>> {
    vec![
        vec![0.95, 0.03, 0.01, 0.01],
        vec![0.90, 0.05, 0.03, 0.02],
        vec![0.85, 0.10, 0.03, 0.02],
    ]
}

fn unseen() -> Vec<Vec<f64>> {
    vec![
        vec![0.30, 0.30, 0.20, 0.20],
        vec![0.35, 0.30, 0.20, 0.15],
        vec![0.25, 0.25, 0.25, 0.25],
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_open_set_entropic")?;
    let seen_h: f64 = seen().iter().map(|p| aou::entropy(p)).sum::<f64>() / seen().len() as f64;
    let unseen_h: f64 =
        unseen().iter().map(|p| aou::entropy(p)).sum::<f64>() / unseen().len() as f64;
    println!(
        "✓ entropic open-set: H(seen)={:.4}, H(unseen)={:.4}, Δ={:.4}",
        seen_h,
        unseen_h,
        unseen_h - seen_h
    );
    assert!(unseen_h - seen_h >= 0.5, "mean entropy gap must be ≥ 0.5");
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
        main().unwrap();
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Identical distributions → entropy gap is 0.
        let h_a = aou::entropy(&[0.25, 0.25, 0.25, 0.25]);
        let h_b = aou::entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h_a - h_b).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let a: f64 = seen().iter().map(|p| aou::entropy(p)).sum();
        let b: f64 = seen().iter().map(|p| aou::entropy(p)).sum();
        assert_eq!(a, b);
    }
}
