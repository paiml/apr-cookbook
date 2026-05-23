//! # Monitoring Disk IO Pressure Detector
//!
//! High disk IO can stall inference if the model paging tier or log
//! tier is saturated. Pressure score = (read_bytes + write_bytes) /
//! sustained_throughput. Verdict: low/elevated/saturated.
//!
//! Demonstrates the **MON.42** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Linux PSI (Pressure Stall Information) interface.
//!
//! Run with: cargo run --example monitor_disk_io_pressure
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IoVerdict {
    Low { utilization_pct: f64 },
    Elevated { utilization_pct: f64 },
    Saturated { utilization_pct: f64 },
    InvalidConfig,
}

pub fn detect(
    read_bytes_per_sec: u64,
    write_bytes_per_sec: u64,
    sustained_throughput_bytes_per_sec: u64,
) -> IoVerdict {
    if sustained_throughput_bytes_per_sec == 0 {
        return IoVerdict::InvalidConfig;
    }
    let total = read_bytes_per_sec.saturating_add(write_bytes_per_sec);
    let utilization_pct = (total as f64 / sustained_throughput_bytes_per_sec as f64) * 100.0;
    if utilization_pct >= 90.0 {
        IoVerdict::Saturated { utilization_pct }
    } else if utilization_pct >= 60.0 {
        IoVerdict::Elevated { utilization_pct }
    } else {
        IoVerdict::Low { utilization_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_disk_io_pressure")?;

    println!("low: {:?}", detect(10_000_000, 5_000_000, 100_000_000));
    println!(
        "elevated: {:?}",
        detect(40_000_000, 20_000_000, 100_000_000)
    );
    println!(
        "saturated: {:?}",
        detect(60_000_000, 30_000_000, 100_000_000)
    );
    println!("invalid: {:?}", detect(100, 100, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_below_60_pct() {
        let v = detect(10_000_000, 5_000_000, 100_000_000);
        assert!(matches!(v, IoVerdict::Low { .. }));
    }

    #[test]
    fn elevated_60_to_90_pct() {
        let v = detect(40_000_000, 20_000_000, 100_000_000);
        assert!(matches!(v, IoVerdict::Elevated { .. }));
    }

    #[test]
    fn saturated_above_90_pct() {
        let v = detect(60_000_000, 30_000_000, 100_000_000);
        assert!(matches!(v, IoVerdict::Saturated { .. }));
    }

    #[test]
    fn zero_throughput_invalid() {
        assert_eq!(detect(100, 100, 0), IoVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_60_elevated() {
        let v = detect(60_000_000, 0, 100_000_000);
        assert!(matches!(v, IoVerdict::Elevated { .. }));
    }

    #[test]
    fn boundary_at_90_saturated() {
        let v = detect(90_000_000, 0, 100_000_000);
        assert!(matches!(v, IoVerdict::Saturated { .. }));
    }

    #[test]
    fn read_only_saturation() {
        let v = detect(100_000_000, 0, 100_000_000);
        assert!(matches!(v, IoVerdict::Saturated { .. }));
    }

    #[test]
    fn write_only_saturation() {
        let v = detect(0, 100_000_000, 100_000_000);
        assert!(matches!(v, IoVerdict::Saturated { .. }));
    }

    #[test]
    fn util_value_correct() {
        let v = detect(50_000_000, 25_000_000, 100_000_000);
        if let IoVerdict::Elevated { utilization_pct } = v {
            assert!((utilization_pct - 75.0).abs() < 1e-6);
        }
    }

    #[test]
    fn over_capacity_capped_logically() {
        let v = detect(200_000_000, 0, 100_000_000);
        if let IoVerdict::Saturated { utilization_pct } = v {
            // Even if reads exceed sustained, returns 200%.
            assert!(utilization_pct >= 100.0);
        }
    }

    #[test]
    fn deterministic() {
        let a = detect(50_000_000, 25_000_000, 100_000_000);
        let b = detect(50_000_000, 25_000_000, 100_000_000);
        assert_eq!(a, b);
    }
}
