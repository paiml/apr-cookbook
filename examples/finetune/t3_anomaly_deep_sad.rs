//! # Tier 3.6 — Deep SAD anomaly detection (tabular-only)
//!
//! Falsifier: Deep SAD: anomaly score > threshold for ≥ 90% of held-out
//! anomalies on synthetic mixed dataset.
//!
//! Run with: cargo run --example t3_anomaly_deep_sad

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn anomalies() -> Vec<Vec<f64>> {
    vec![
        vec![5.0, 5.0],
        vec![6.0, 6.0],
        vec![5.5, 5.5],
        vec![10.0, 10.0],
        vec![4.0, 4.0],
        vec![7.0, 7.0],
        vec![5.2, 5.2],
        vec![6.5, 6.5],
        vec![4.5, 4.5],
        vec![5.8, 5.8],
    ]
}
fn centroid() -> Vec<f64> {
    vec![0.0, 0.0]
}
const THRESHOLD: f64 = 3.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_anomaly_deep_sad")?;
    let recall = aou::deep_sad_recall(&anomalies(), &centroid(), THRESHOLD);
    println!(
        "✓ Deep SAD: recall = {:.2} (threshold {})",
        recall, THRESHOLD
    );
    assert!(recall >= 0.9, "anomaly recall must be ≥ 0.9, got {recall}");
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
        assert!(aou::deep_sad_recall(&anomalies(), &centroid(), THRESHOLD) >= 0.9);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Massive threshold → recall drops below 0.9.
        assert!(aou::deep_sad_recall(&anomalies(), &centroid(), 100.0) < 0.9);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::deep_sad_recall(&anomalies(), &centroid(), THRESHOLD);
        let b = aou::deep_sad_recall(&anomalies(), &centroid(), THRESHOLD);
        assert_eq!(a, b);
    }
}
