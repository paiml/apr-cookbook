//! # Monitoring Log Volume Anomaly
//!
//! Sudden spike in log volume often signals a runaway loop, retry
//! storm, or panic burst. Detector: current_minute_lines vs
//! baseline_p95 → Normal/Spike/Quiet.
//!
//! Demonstrates the **MON.47** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ELK / Datadog log-volume anomaly detection.
//!
//! Run with: cargo run --example monitor_log_volume_anomaly
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LogVerdict {
    Normal { ratio: f64 },
    Spike { ratio: f64 },
    Quiet { ratio: f64 },
    InvalidConfig,
}

pub fn check(current_lines_per_min: u64, baseline_p95_lines_per_min: u64) -> LogVerdict {
    if baseline_p95_lines_per_min == 0 {
        return LogVerdict::InvalidConfig;
    }
    let ratio = current_lines_per_min as f64 / baseline_p95_lines_per_min as f64;
    if ratio >= 3.0 {
        LogVerdict::Spike { ratio }
    } else if ratio <= 0.10 {
        LogVerdict::Quiet { ratio }
    } else {
        LogVerdict::Normal { ratio }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_log_volume_anomaly")?;

    println!("normal: {:?}", check(1_000, 1_200));
    println!("spike: {:?}", check(5_000, 1_000));
    println!("quiet: {:?}", check(50, 1_000));
    println!("invalid: {:?}", check(100, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_at_baseline() {
        let v = check(1_000, 1_000);
        assert!(matches!(v, LogVerdict::Normal { .. }));
    }

    #[test]
    fn spike_at_3x() {
        let v = check(3_000, 1_000);
        assert!(matches!(v, LogVerdict::Spike { .. }));
    }

    #[test]
    fn spike_at_5x() {
        let v = check(5_000, 1_000);
        assert!(matches!(v, LogVerdict::Spike { .. }));
    }

    #[test]
    fn quiet_at_10_pct() {
        let v = check(100, 1_000);
        assert!(matches!(v, LogVerdict::Quiet { .. }));
    }

    #[test]
    fn quiet_at_zero() {
        let v = check(0, 1_000);
        assert!(matches!(v, LogVerdict::Quiet { .. }));
    }

    #[test]
    fn zero_baseline_invalid() {
        assert_eq!(check(100, 0), LogVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_10_pct_quiet() {
        let v = check(100, 1_000);
        assert!(matches!(v, LogVerdict::Quiet { .. }));
    }

    #[test]
    fn just_above_10_pct_normal() {
        let v = check(101, 1_000);
        assert!(matches!(v, LogVerdict::Normal { .. }));
    }

    #[test]
    fn just_below_3x_normal() {
        let v = check(2_999, 1_000);
        assert!(matches!(v, LogVerdict::Normal { .. }));
    }

    #[test]
    fn ratio_value_correct() {
        let v = check(2_500, 1_000);
        if let LogVerdict::Normal { ratio } = v {
            assert!((ratio - 2.5).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(2_500, 1_000);
        let b = check(2_500, 1_000);
        assert_eq!(a, b);
    }
}
