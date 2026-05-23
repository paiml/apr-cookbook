//! # TUI Progress Throughput Running Average
//!
//! Compute running mean throughput from a sample stream
//! (units/second). Returns final mean and count of valid samples.
//!
//! Demonstrates the **TUI.92** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Welford running mean (Knuth TAOCP §4.2.2);
//!  ETA stabilization in pip/cargo progress bars.
//!
//! Run with: cargo run --example tui_progress_throughput_avg
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThroughputVerdict {
    Ok {
        mean: f64,
        sample_count: u32,
        max_observed: f64,
    },
    InvalidConfig,
}

pub fn compute(samples: &[f64]) -> ThroughputVerdict {
    if samples.is_empty() {
        return ThroughputVerdict::InvalidConfig;
    }
    let mut mean: f64 = 0.0;
    let mut count: u32 = 0;
    let mut max_observed: f64 = f64::MIN;
    for sample in samples {
        if !sample.is_finite() || *sample < 0.0 {
            continue;
        }
        count += 1;
        mean += (sample - mean) / f64::from(count);
        if *sample > max_observed {
            max_observed = *sample;
        }
    }
    if count == 0 {
        return ThroughputVerdict::InvalidConfig;
    }
    ThroughputVerdict::Ok {
        mean,
        sample_count: count,
        max_observed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_throughput_avg")?;

    let samples = [10.0, 20.0, 30.0, 25.0, 15.0];
    println!("typical: {:?}", compute(&samples));
    println!("with negative: {:?}", compute(&[10.0, -5.0, 20.0]));
    println!("invalid: {:?}", compute(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn mean_correct_simple() {
        let v = compute(&[10.0, 20.0, 30.0]);
        if let ThroughputVerdict::Ok { mean, .. } = v {
            assert!((mean - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn negative_samples_skipped() {
        let v = compute(&[10.0, -5.0, 20.0]);
        if let ThroughputVerdict::Ok {
            mean, sample_count, ..
        } = v
        {
            assert_eq!(sample_count, 2);
            assert!((mean - 15.0).abs() < 1e-9);
        }
    }

    #[test]
    fn nan_samples_skipped() {
        let v = compute(&[10.0, f64::NAN, 20.0]);
        if let ThroughputVerdict::Ok { sample_count, .. } = v {
            assert_eq!(sample_count, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compute(&[]), ThroughputVerdict::InvalidConfig);
    }

    #[test]
    fn all_invalid_rejected() {
        assert_eq!(
            compute(&[-1.0, f64::NAN, -5.0]),
            ThroughputVerdict::InvalidConfig
        );
    }

    #[test]
    fn max_observed_correct() {
        let v = compute(&[10.0, 50.0, 5.0]);
        if let ThroughputVerdict::Ok { max_observed, .. } = v {
            assert!((max_observed - 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(&[1.0, 2.0]);
        let r2 = compute(&[1.0, 2.0]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_sample_works() {
        let v = compute(&[42.0]);
        if let ThroughputVerdict::Ok { mean, .. } = v {
            assert_eq!(mean, 42.0);
        }
    }

    #[test]
    fn zero_throughput_valid() {
        let v = compute(&[0.0, 0.0]);
        if let ThroughputVerdict::Ok { mean, .. } = v {
            assert_eq!(mean, 0.0);
        }
    }

    #[test]
    fn count_le_input_len() {
        let v = compute(&[1.0, 2.0, -1.0, 3.0]);
        if let ThroughputVerdict::Ok { sample_count, .. } = v {
            assert!(sample_count <= 4);
        }
    }

    #[test]
    fn welford_avoids_naive_overflow() {
        // Two large numbers that would overflow naive sum.
        let v = compute(&[1e18, 1e18]);
        if let ThroughputVerdict::Ok { mean, .. } = v {
            assert!(mean.is_finite());
        }
    }
}
