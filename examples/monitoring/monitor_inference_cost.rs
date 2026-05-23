//! # Monitoring Per-Request Inference Cost
//!
//! Cost = gpu_seconds × $/gpu_sec + bytes_out × $/GB.
//! Plus tier classifier: Cheap (< $0.001), Normal (< $0.01),
//! Expensive (< $0.10), VeryExpensive (≥ $0.10).
//!
//! Demonstrates the **MON.19** recipe for PMAT-140 (monitoring round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS/GCP/Azure GPU pricing (2024 Q4 representative rates).
//!
//! Run with: cargo run --example monitor_inference_cost
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostTier {
    Cheap,
    Normal,
    Expensive,
    VeryExpensive,
}

#[derive(Debug, PartialEq)]
pub enum CostVerdict {
    Ok { cost_usd: f64, tier: CostTier },
    InvalidGpuRate,
    InvalidEgressRate,
    InvalidUsage,
}

pub fn calc(
    gpu_seconds: f64,
    gpu_usd_per_sec: f64,
    bytes_out: u64,
    egress_usd_per_gb: f64,
) -> CostVerdict {
    if !gpu_seconds.is_finite() || gpu_seconds < 0.0 {
        return CostVerdict::InvalidUsage;
    }
    if !gpu_usd_per_sec.is_finite() || gpu_usd_per_sec < 0.0 {
        return CostVerdict::InvalidGpuRate;
    }
    if !egress_usd_per_gb.is_finite() || egress_usd_per_gb < 0.0 {
        return CostVerdict::InvalidEgressRate;
    }
    let gpu_cost = gpu_seconds * gpu_usd_per_sec;
    let egress_gb = bytes_out as f64 / 1_073_741_824.0;
    let egress_cost = egress_gb * egress_usd_per_gb;
    let cost_usd = gpu_cost + egress_cost;
    let tier = if cost_usd < 0.001 {
        CostTier::Cheap
    } else if cost_usd < 0.01 {
        CostTier::Normal
    } else if cost_usd < 0.10 {
        CostTier::Expensive
    } else {
        CostTier::VeryExpensive
    };
    CostVerdict::Ok { cost_usd, tier }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_inference_cost")?;

    // A100 ≈ $0.001/sec, egress ≈ $0.09/GB.
    println!("0.5s + 1KB: {:?}", calc(0.5, 0.001, 1024, 0.09));
    println!("5s + 1MB: {:?}", calc(5.0, 0.001, 1_048_576, 0.09));
    println!("60s + 1GB: {:?}", calc(60.0, 0.001, 1_073_741_824, 0.09));
    println!("invalid neg: {:?}", calc(-1.0, 0.001, 0, 0.09));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_request_cheap() {
        let v = calc(0.1, 0.001, 100, 0.09);
        if let CostVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Cheap);
        }
    }

    #[test]
    fn medium_request_normal() {
        // 5s × $0.001 = $0.005 → Normal.
        let v = calc(5.0, 0.001, 0, 0.09);
        if let CostVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Normal);
        }
    }

    #[test]
    fn large_egress_expensive() {
        // 1s × $0.001 = $0.001 + 1 GB × $0.09 = $0.091 → Expensive.
        let v = calc(1.0, 0.001, 1_073_741_824, 0.09);
        if let CostVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Expensive);
        }
    }

    #[test]
    fn very_large_request_very_expensive() {
        // 60s × $0.01/sec = $0.60 → VeryExpensive.
        let v = calc(60.0, 0.01, 0, 0.09);
        if let CostVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::VeryExpensive);
        }
    }

    #[test]
    fn invalid_neg_gpu_seconds_rejected() {
        assert_eq!(calc(-1.0, 0.001, 0, 0.09), CostVerdict::InvalidUsage);
    }

    #[test]
    fn invalid_neg_gpu_rate_rejected() {
        assert_eq!(calc(1.0, -0.01, 0, 0.09), CostVerdict::InvalidGpuRate);
    }

    #[test]
    fn invalid_neg_egress_rate_rejected() {
        assert_eq!(calc(1.0, 0.01, 0, -0.01), CostVerdict::InvalidEgressRate);
    }

    #[test]
    fn nan_input_rejected() {
        assert_eq!(calc(f64::NAN, 0.01, 0, 0.09), CostVerdict::InvalidUsage);
    }

    #[test]
    fn zero_usage_zero_cost() {
        let v = calc(0.0, 0.01, 0, 0.09);
        if let CostVerdict::Ok { cost_usd, tier } = v {
            assert!(cost_usd.abs() < 1e-12);
            assert_eq!(tier, CostTier::Cheap);
        }
    }

    #[test]
    fn cost_proportional_to_seconds() {
        let v1 = calc(1.0, 0.001, 0, 0.09);
        let v2 = calc(10.0, 0.001, 0, 0.09);
        if let (CostVerdict::Ok { cost_usd: c1, .. }, CostVerdict::Ok { cost_usd: c2, .. }) =
            (v1, v2)
        {
            assert!((c2 / c1 - 10.0).abs() < 1e-9);
        }
    }
}
