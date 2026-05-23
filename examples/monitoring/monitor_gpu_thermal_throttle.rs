//! # Monitoring GPU Thermal-Throttle Detector
//!
//! GPUs throttle clock speed when temperature exceeds the thermal
//! envelope. This recipe detects throttling by checking temperature
//! and observed clock speed against rated clock.
//!
//! Demonstrates the **MON.39** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA NVML thermal throttling fields.
//!
//! Run with: cargo run --example monitor_gpu_thermal_throttle
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThermalVerdict {
    Healthy,
    ThrottledClock { clock_loss_pct: f64 },
    OverTemp { temp_c: f64 },
    Critical { temp_c: f64, clock_loss_pct: f64 },
    InvalidReading,
}

pub fn detect(
    temp_c: f64,
    observed_clock_mhz: f64,
    rated_clock_mhz: f64,
    throttle_temp_c: f64,
) -> ThermalVerdict {
    if !temp_c.is_finite()
        || !observed_clock_mhz.is_finite()
        || !rated_clock_mhz.is_finite()
        || rated_clock_mhz <= 0.0
        || observed_clock_mhz < 0.0
        || throttle_temp_c <= 0.0
    {
        return ThermalVerdict::InvalidReading;
    }
    let clock_loss_pct = ((rated_clock_mhz - observed_clock_mhz) / rated_clock_mhz) * 100.0;
    let throttling = clock_loss_pct > 5.0;
    let over_temp = temp_c >= throttle_temp_c;
    match (over_temp, throttling) {
        (true, true) => ThermalVerdict::Critical {
            temp_c,
            clock_loss_pct,
        },
        (true, false) => ThermalVerdict::OverTemp { temp_c },
        (false, true) => ThermalVerdict::ThrottledClock { clock_loss_pct },
        (false, false) => ThermalVerdict::Healthy,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_gpu_thermal_throttle")?;

    println!("healthy: {:?}", detect(60.0, 1500.0, 1500.0, 85.0));
    println!("throttled: {:?}", detect(60.0, 1200.0, 1500.0, 85.0));
    println!("over temp: {:?}", detect(90.0, 1500.0, 1500.0, 85.0));
    println!("critical: {:?}", detect(90.0, 1000.0, 1500.0, 85.0));
    println!("invalid: {:?}", detect(60.0, 1500.0, -1.0, 85.0));
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
    fn healthy_at_normal_load() {
        let v = detect(60.0, 1500.0, 1500.0, 85.0);
        assert_eq!(v, ThermalVerdict::Healthy);
    }

    #[test]
    fn throttle_when_clock_low() {
        let v = detect(60.0, 1200.0, 1500.0, 85.0);
        assert!(matches!(v, ThermalVerdict::ThrottledClock { .. }));
    }

    #[test]
    fn over_temp_alone() {
        let v = detect(90.0, 1500.0, 1500.0, 85.0);
        assert!(matches!(v, ThermalVerdict::OverTemp { .. }));
    }

    #[test]
    fn critical_when_both() {
        let v = detect(90.0, 1000.0, 1500.0, 85.0);
        assert!(matches!(v, ThermalVerdict::Critical { .. }));
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(
            detect(f64::NAN, 1500.0, 1500.0, 85.0),
            ThermalVerdict::InvalidReading
        );
    }

    #[test]
    fn zero_rated_clock_rejected() {
        assert_eq!(
            detect(60.0, 1500.0, 0.0, 85.0),
            ThermalVerdict::InvalidReading
        );
    }

    #[test]
    fn zero_throttle_temp_rejected() {
        assert_eq!(
            detect(60.0, 1500.0, 1500.0, 0.0),
            ThermalVerdict::InvalidReading
        );
    }

    #[test]
    fn small_clock_loss_under_5pct_healthy() {
        // 4% loss → healthy.
        let v = detect(60.0, 1440.0, 1500.0, 85.0);
        assert_eq!(v, ThermalVerdict::Healthy);
    }

    #[test]
    fn boundary_at_throttle_temp() {
        // Exactly at throttle temp → over_temp.
        let v = detect(85.0, 1500.0, 1500.0, 85.0);
        assert!(matches!(v, ThermalVerdict::OverTemp { .. }));
    }

    #[test]
    fn clock_loss_value_returned() {
        let v = detect(60.0, 1200.0, 1500.0, 85.0);
        if let ThermalVerdict::ThrottledClock { clock_loss_pct } = v {
            // (1500-1200)/1500 = 20%.
            assert!((clock_loss_pct - 20.0).abs() < 1e-6);
        }
    }

    #[test]
    fn deterministic() {
        let a = detect(60.0, 1200.0, 1500.0, 85.0);
        let b = detect(60.0, 1200.0, 1500.0, 85.0);
        assert_eq!(a, b);
    }
}
