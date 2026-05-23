//! # Distributed Gradient Compression Strategy
//!
//! Reduce all-reduce bandwidth by compressing gradients:
//!   None: full fp32 (1× bandwidth, lossless)
//!   Fp16: 2× compression, minimal accuracy loss
//!   TopK (1%): 100× compression, sparse, requires error-feedback
//!   SignSGD: 32× compression (1 bit per grad), aggressive
//!
//! Picker: trade-off bandwidth vs. accuracy.
//!
//! Demonstrates the **DIST.17** recipe for PMAT-150 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PowerSGD (Vogels et al., 2019) + DGC (Lin et al., 2018).
//!
//! Run with: cargo run --example distributed_grad_compression
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    None,
    Fp16,
    TopK { sparsity_pct: u32 },
    SignSGD,
}

#[derive(Debug, PartialEq)]
pub enum GradVerdict {
    Ok {
        strategy: CompressionStrategy,
        compression_ratio: f64,
        accuracy_impact_pct: f64,
    },
    InvalidBandwidth,
}

pub fn pick(
    available_bandwidth_gbps: f64,
    target_throughput_gbps: f64,
    accuracy_critical: bool,
) -> GradVerdict {
    if !available_bandwidth_gbps.is_finite()
        || !target_throughput_gbps.is_finite()
        || available_bandwidth_gbps <= 0.0
        || target_throughput_gbps <= 0.0
    {
        return GradVerdict::InvalidBandwidth;
    }
    let needed_ratio = target_throughput_gbps / available_bandwidth_gbps;
    let strategy = if needed_ratio <= 1.0 {
        CompressionStrategy::None
    } else if needed_ratio <= 2.0 {
        CompressionStrategy::Fp16
    } else if accuracy_critical {
        CompressionStrategy::TopK { sparsity_pct: 1 }
    } else if needed_ratio <= 32.0 {
        CompressionStrategy::SignSGD
    } else {
        CompressionStrategy::TopK { sparsity_pct: 1 }
    };
    let (compression_ratio, accuracy_impact_pct) = match strategy {
        CompressionStrategy::None => (1.0, 0.0),
        CompressionStrategy::Fp16 => (2.0, 0.5),
        CompressionStrategy::TopK { .. } => (100.0, 1.5),
        CompressionStrategy::SignSGD => (32.0, 3.0),
    };
    GradVerdict::Ok {
        strategy,
        compression_ratio,
        accuracy_impact_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_grad_compression")?;

    println!("ample bw: {:?}", pick(100.0, 50.0, false));
    println!("modest 2x: {:?}", pick(50.0, 80.0, false));
    println!("aggressive: {:?}", pick(10.0, 200.0, false));
    println!("aggressive accuracy: {:?}", pick(10.0, 200.0, true));
    println!("invalid: {:?}", pick(0.0, 100.0, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ample_bandwidth_no_compression() {
        let v = pick(100.0, 50.0, false);
        if let GradVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, CompressionStrategy::None);
        }
    }

    #[test]
    fn modest_picks_fp16() {
        let v = pick(50.0, 80.0, false);
        if let GradVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, CompressionStrategy::Fp16);
        }
    }

    #[test]
    fn aggressive_signsgd_default() {
        let v = pick(10.0, 200.0, false);
        if let GradVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, CompressionStrategy::SignSGD);
        }
    }

    #[test]
    fn aggressive_accuracy_topk() {
        let v = pick(10.0, 200.0, true);
        if let GradVerdict::Ok { strategy, .. } = v {
            assert!(matches!(strategy, CompressionStrategy::TopK { .. }));
        }
    }

    #[test]
    fn invalid_zero_bandwidth() {
        assert_eq!(pick(0.0, 100.0, false), GradVerdict::InvalidBandwidth);
    }

    #[test]
    fn invalid_negative_target() {
        assert_eq!(pick(50.0, -10.0, false), GradVerdict::InvalidBandwidth);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(pick(f64::NAN, 100.0, false), GradVerdict::InvalidBandwidth);
    }

    #[test]
    fn higher_compression_higher_impact() {
        let v_fp16 = pick(50.0, 80.0, false);
        let v_topk = pick(1.0, 200.0, true);
        if let (
            GradVerdict::Ok {
                accuracy_impact_pct: f,
                ..
            },
            GradVerdict::Ok {
                accuracy_impact_pct: t,
                ..
            },
        ) = (v_fp16, v_topk)
        {
            assert!(t > f);
        }
    }

    #[test]
    fn compression_ratio_at_least_one() {
        for (avail, tgt) in [(100.0, 50.0), (50.0, 80.0), (10.0, 200.0)] {
            if let GradVerdict::Ok {
                compression_ratio, ..
            } = pick(avail, tgt, false)
            {
                assert!(compression_ratio >= 1.0);
            }
        }
    }

    #[test]
    fn boundary_at_one_no_compression() {
        let v = pick(100.0, 100.0, false);
        if let GradVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, CompressionStrategy::None);
        }
    }
}
