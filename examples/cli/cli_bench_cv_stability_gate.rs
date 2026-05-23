//! # apr bench --cv-gate — Coefficient of Variation Stability Gate
//!
//! `apr bench` reports per-iteration latency. Coefficient of variation
//! (CV = σ/μ) measures stability: under 5% is stable, 5-15% is noisy,
//! over 15% is noise-dominated (rerun with longer warmup or quieter
//! host). This recipe builds the gate for CI rejection of noisy benchmarks.
//!
//! Demonstrates the **BENCH.4** recipe for PMAT-118 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender BENCH-001 + Gregg 2016 (Systems Performance)
//!
//! Run with: cargo run --example cli_bench_cv_stability_gate
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StabilityTier {
    Stable,
    Noisy,
    NoiseDominated,
    InvalidSamples,
}

const STABLE_CAP: f64 = 0.05;
const NOISY_CAP: f64 = 0.15;

pub fn coefficient_of_variation(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if !mean.is_finite() || mean == 0.0 {
        return None;
    }
    let variance: f64 =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let std = variance.sqrt();
    Some(std / mean.abs())
}

pub fn classify(samples: &[f64]) -> StabilityTier {
    let Some(cv) = coefficient_of_variation(samples) else {
        return StabilityTier::InvalidSamples;
    };
    if cv < STABLE_CAP {
        StabilityTier::Stable
    } else if cv < NOISY_CAP {
        StabilityTier::Noisy
    } else {
        StabilityTier::NoiseDominated
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_cv_stability_gate")?;

    let stable = vec![10.0, 10.1, 9.9, 10.05, 9.95];
    let noisy = vec![10.0, 11.0, 9.0, 12.0, 8.0];
    let chaotic = vec![10.0, 50.0, 5.0, 100.0, 2.0];
    let single = vec![10.0];

    for (label, samples) in [
        ("stable", &stable),
        ("noisy", &noisy),
        ("chaotic", &chaotic),
        ("single", &single),
    ] {
        println!(
            "{label:>10}  cv={:?}  →  {:?}",
            coefficient_of_variation(samples),
            classify(samples)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stable_samples_classified_stable() {
        // CV < 5% → Stable.
        let samples = vec![10.0, 10.1, 9.9, 10.05, 9.95];
        assert_eq!(classify(&samples), StabilityTier::Stable);
    }

    #[test]
    fn moderate_noise_classified_noisy() {
        let samples = vec![10.0, 11.0, 9.0, 11.5, 8.5];
        assert_eq!(classify(&samples), StabilityTier::Noisy);
    }

    #[test]
    fn high_noise_classified_dominated() {
        let samples = vec![10.0, 50.0, 5.0, 100.0, 2.0];
        assert_eq!(classify(&samples), StabilityTier::NoiseDominated);
    }

    #[test]
    fn single_sample_invalid() {
        assert_eq!(classify(&[10.0]), StabilityTier::InvalidSamples);
    }

    #[test]
    fn empty_invalid() {
        assert_eq!(classify(&[]), StabilityTier::InvalidSamples);
    }

    #[test]
    fn zero_mean_invalid() {
        // CV undefined when mean = 0.
        assert!(coefficient_of_variation(&[1.0, -1.0]).is_none());
    }

    #[test]
    fn cv_zero_for_constant_samples() {
        let cv = coefficient_of_variation(&[5.0, 5.0, 5.0, 5.0]).unwrap();
        assert!(cv.abs() < 1e-12);
    }

    #[test]
    fn cv_invariant_to_scale() {
        // Multiplying all samples by a constant preserves CV.
        let cv1 = coefficient_of_variation(&[1.0, 2.0, 3.0]).unwrap();
        let cv2 = coefficient_of_variation(&[100.0, 200.0, 300.0]).unwrap();
        assert!((cv1 - cv2).abs() < 1e-9);
    }

    #[test]
    fn negative_mean_uses_abs() {
        // CV uses |mean|; works with negative measurements too.
        let cv = coefficient_of_variation(&[-10.0, -10.1, -9.9]);
        assert!(cv.is_some() && cv.unwrap() < 0.05);
    }
}
