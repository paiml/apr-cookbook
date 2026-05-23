//! # Monitoring OOM Predictor
//!
//! Predict time-to-OOM by linear extrapolation: given current memory
//! and growth rate (bytes/sec), estimate seconds until cap. Useful for
//! pre-emptive eviction or scaling.
//!
//! Demonstrates the **MON.46** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cgroup memory pressure prediction (Linux PSI).
//!
//! Run with: cargo run --example monitor_oom_predict
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OomVerdict {
    Stable,
    Imminent { seconds_to_oom: f64 },
    Soon { seconds_to_oom: f64 },
    Eventual { seconds_to_oom: f64 },
    AlreadyOver,
    InvalidConfig,
}

pub fn predict(current_bytes: u64, cap_bytes: u64, growth_bytes_per_sec: f64) -> OomVerdict {
    if cap_bytes == 0 || !growth_bytes_per_sec.is_finite() {
        return OomVerdict::InvalidConfig;
    }
    if current_bytes >= cap_bytes {
        return OomVerdict::AlreadyOver;
    }
    if growth_bytes_per_sec <= 0.0 {
        return OomVerdict::Stable;
    }
    let remaining = (cap_bytes - current_bytes) as f64;
    let seconds_to_oom = remaining / growth_bytes_per_sec;
    if seconds_to_oom < 60.0 {
        OomVerdict::Imminent { seconds_to_oom }
    } else if seconds_to_oom < 600.0 {
        OomVerdict::Soon { seconds_to_oom }
    } else {
        OomVerdict::Eventual { seconds_to_oom }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_oom_predict")?;

    println!("stable: {:?}", predict(8_000_000_000, 16_000_000_000, 0.0));
    println!(
        "imminent: {:?}",
        predict(15_500_000_000, 16_000_000_000, 50_000_000.0)
    );
    println!(
        "soon: {:?}",
        predict(13_000_000_000, 16_000_000_000, 10_000_000.0)
    );
    println!(
        "eventual: {:?}",
        predict(8_000_000_000, 16_000_000_000, 1_000_000.0)
    );
    println!("over: {:?}", predict(20_000_000_000, 16_000_000_000, 1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stable_when_no_growth() {
        let v = predict(8_000_000_000, 16_000_000_000, 0.0);
        assert_eq!(v, OomVerdict::Stable);
    }

    #[test]
    fn negative_growth_stable() {
        let v = predict(8_000_000_000, 16_000_000_000, -1.0);
        assert_eq!(v, OomVerdict::Stable);
    }

    #[test]
    fn imminent_under_60s() {
        let v = predict(15_500_000_000, 16_000_000_000, 50_000_000.0);
        assert!(matches!(v, OomVerdict::Imminent { .. }));
    }

    #[test]
    fn soon_60_to_600s() {
        let v = predict(13_000_000_000, 16_000_000_000, 10_000_000.0);
        assert!(matches!(v, OomVerdict::Soon { .. }));
    }

    #[test]
    fn eventual_over_600s() {
        let v = predict(8_000_000_000, 16_000_000_000, 1_000_000.0);
        assert!(matches!(v, OomVerdict::Eventual { .. }));
    }

    #[test]
    fn already_over_classified() {
        let v = predict(20_000_000_000, 16_000_000_000, 1.0);
        assert_eq!(v, OomVerdict::AlreadyOver);
    }

    #[test]
    fn equal_to_cap_already_over() {
        let v = predict(16_000_000_000, 16_000_000_000, 1.0);
        assert_eq!(v, OomVerdict::AlreadyOver);
    }

    #[test]
    fn zero_cap_invalid() {
        assert_eq!(predict(100, 0, 1.0), OomVerdict::InvalidConfig);
    }

    #[test]
    fn nan_growth_invalid() {
        assert_eq!(predict(100, 1000, f64::NAN), OomVerdict::InvalidConfig);
    }

    #[test]
    fn seconds_value_correct() {
        // 1MB remaining, 1B/s growth → 1e6 seconds.
        let v = predict(0, 1_000_000, 1.0);
        if let OomVerdict::Eventual { seconds_to_oom } = v {
            assert!((seconds_to_oom - 1_000_000.0).abs() < 1.0);
        }
    }

    #[test]
    fn deterministic() {
        let a = predict(8_000_000_000, 16_000_000_000, 1_000_000.0);
        let b = predict(8_000_000_000, 16_000_000_000, 1_000_000.0);
        assert_eq!(a, b);
    }
}
