//! # Analysis P99 Throughput SLA Gate
//!
//! Throughput SLA: p99(latency) ≤ target. Stream-process latency
//! samples + emit pass/fail with the actual p99. Includes a sample-
//! sufficiency check (≥ 100 samples for stable p99 estimate).
//!
//! Demonstrates the **ANL.53** recipe for PMAT-131 (analysis coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gil Tene (2015). Understanding HDR Histogram.
//!
//! Run with: cargo run --example analysis_p99_throughput_gate
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_SAMPLES: usize = 100;

#[derive(Debug, PartialEq)]
pub enum SlaVerdict {
    Pass { p99_ms: f64, target_ms: f64 },
    Fail { p99_ms: f64, target_ms: f64 },
    InsufficientSamples { got: usize, need: usize },
    InvalidTarget,
}

pub fn p99(samples_ms: &[f64]) -> Option<f64> {
    if samples_ms.is_empty() {
        return None;
    }
    if samples_ms.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut sorted: Vec<f64> = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64) * 0.99).ceil() as usize - 1;
    Some(sorted[idx.min(sorted.len() - 1)])
}

pub fn gate(samples_ms: &[f64], target_ms: f64) -> SlaVerdict {
    if !target_ms.is_finite() || target_ms <= 0.0 {
        return SlaVerdict::InvalidTarget;
    }
    if samples_ms.len() < MIN_SAMPLES {
        return SlaVerdict::InsufficientSamples {
            got: samples_ms.len(),
            need: MIN_SAMPLES,
        };
    }
    let Some(p) = p99(samples_ms) else {
        return SlaVerdict::InvalidTarget;
    };
    if p <= target_ms {
        SlaVerdict::Pass {
            p99_ms: p,
            target_ms,
        }
    } else {
        SlaVerdict::Fail {
            p99_ms: p,
            target_ms,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_p99_throughput_gate")?;

    let mut samples: Vec<f64> = (1..=1000).map(|i| i as f64 / 10.0).collect();
    println!("p99: {:?}", p99(&samples));
    println!("gate vs 100ms: {:?}", gate(&samples, 100.0));

    samples.push(500.0); // outlier
    println!("with outlier vs 100ms: {:?}", gate(&samples, 100.0));

    println!("too few: {:?}", gate(&[1.0, 2.0], 100.0));
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
    fn p99_basic_correctness() {
        let s: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        // p99 of 1..=100 → index ceil(99) - 1 = 98 → value 99.
        assert_eq!(p99(&s), Some(99.0));
    }

    #[test]
    fn p99_empty_returns_none() {
        assert!(p99(&[]).is_none());
    }

    #[test]
    fn p99_nan_returns_none() {
        assert!(p99(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn pass_when_under_target() {
        let s: Vec<f64> = (1..=200).map(|i| i as f64).collect();
        let v = gate(&s, 250.0);
        assert!(matches!(v, SlaVerdict::Pass { .. }));
    }

    #[test]
    fn fail_when_over_target() {
        let s: Vec<f64> = (1..=200).map(|i| i as f64).collect();
        let v = gate(&s, 50.0);
        assert!(matches!(v, SlaVerdict::Fail { .. }));
    }

    #[test]
    fn too_few_samples_rejected() {
        let v = gate(&[1.0, 2.0, 3.0], 10.0);
        assert!(matches!(v, SlaVerdict::InsufficientSamples { .. }));
    }

    #[test]
    fn at_min_samples_passes_through() {
        let s: Vec<f64> = (1..=MIN_SAMPLES as i32).map(|i| i as f64).collect();
        let v = gate(&s, 1000.0);
        assert!(matches!(v, SlaVerdict::Pass { .. }));
    }

    #[test]
    fn invalid_target_rejected() {
        let s: Vec<f64> = (1..=200).map(|i| i as f64).collect();
        assert_eq!(gate(&s, 0.0), SlaVerdict::InvalidTarget);
        assert_eq!(gate(&s, -10.0), SlaVerdict::InvalidTarget);
    }

    #[test]
    fn unsorted_samples_handled() {
        let s = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        // Sorted: 1, 2, 3, 4, 5; p99 idx = ceil(4.95)-1 = 4 → max = 5.
        assert_eq!(p99(&s), Some(5.0));
    }

    #[test]
    fn boundary_at_target_passes() {
        // p99 == target → pass (≤ inclusive).
        let s: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let v = gate(&s, 99.0);
        assert!(matches!(v, SlaVerdict::Pass { .. }));
    }
}
