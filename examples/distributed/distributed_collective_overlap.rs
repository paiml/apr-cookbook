//! # Distributed Collective-Compute Overlap Picker
//!
//! Overlap layer-N gradient all-reduce with layer-(N+1) backward
//! computation:
//!   compute_ms ≪ allreduce_ms → ExposedComm (compute fast, comm dominates)
//!   roughly equal → Overlap (perfect pipelining)
//!   compute_ms ≫ allreduce_ms → ExposedCompute (comm hidden)
//!
//! Returns expected end-to-end time given overlap.
//!
//! Demonstrates the **DIST.18** recipe for PMAT-150 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch DistributedDataParallel bucket-and-reduce.
//!
//! Run with: cargo run --example distributed_collective_overlap
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapTier {
    ExposedComm,
    Overlap,
    ExposedCompute,
}

#[derive(Debug, PartialEq)]
pub enum OverlapVerdict {
    Ok {
        tier: OverlapTier,
        end_to_end_ms: f64,
        utilization_pct: u32,
    },
    InvalidTiming,
}

pub fn pick(compute_ms: f64, allreduce_ms: f64) -> OverlapVerdict {
    if !compute_ms.is_finite() || !allreduce_ms.is_finite() {
        return OverlapVerdict::InvalidTiming;
    }
    if compute_ms < 0.0 || allreduce_ms < 0.0 {
        return OverlapVerdict::InvalidTiming;
    }
    if compute_ms == 0.0 && allreduce_ms == 0.0 {
        return OverlapVerdict::InvalidTiming;
    }
    let ratio = if allreduce_ms == 0.0 {
        f64::INFINITY
    } else {
        compute_ms / allreduce_ms
    };
    let tier = if ratio < 0.5 {
        OverlapTier::ExposedComm
    } else if ratio <= 2.0 {
        OverlapTier::Overlap
    } else {
        OverlapTier::ExposedCompute
    };
    // End-to-end = max(compute, allreduce) + small overhead.
    let end_to_end_ms = compute_ms.max(allreduce_ms) * 1.05;
    let serial = compute_ms + allreduce_ms;
    let utilization_pct = if serial > 0.0 {
        ((end_to_end_ms / serial) * 100.0).round() as u32
    } else {
        0
    };
    OverlapVerdict::Ok {
        tier,
        end_to_end_ms,
        utilization_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_collective_overlap")?;

    println!("balanced: {:?}", pick(50.0, 50.0));
    println!("comm-bound: {:?}", pick(10.0, 100.0));
    println!("compute-bound: {:?}", pick(100.0, 10.0));
    println!("invalid: {:?}", pick(-1.0, 50.0));
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
    fn balanced_overlap_tier() {
        let v = pick(50.0, 50.0);
        if let OverlapVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, OverlapTier::Overlap);
        }
    }

    #[test]
    fn comm_dominant_exposed_comm() {
        let v = pick(10.0, 100.0);
        if let OverlapVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, OverlapTier::ExposedComm);
        }
    }

    #[test]
    fn compute_dominant_exposed_compute() {
        let v = pick(100.0, 10.0);
        if let OverlapVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, OverlapTier::ExposedCompute);
        }
    }

    #[test]
    fn invalid_negative_rejected() {
        assert_eq!(pick(-1.0, 50.0), OverlapVerdict::InvalidTiming);
    }

    #[test]
    fn invalid_both_zero_rejected() {
        assert_eq!(pick(0.0, 0.0), OverlapVerdict::InvalidTiming);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(pick(f64::NAN, 50.0), OverlapVerdict::InvalidTiming);
    }

    #[test]
    fn end_to_end_at_least_max() {
        let v = pick(50.0, 100.0);
        if let OverlapVerdict::Ok { end_to_end_ms, .. } = v {
            assert!(end_to_end_ms >= 100.0);
        }
    }

    #[test]
    fn end_to_end_less_than_serial() {
        let v = pick(50.0, 50.0);
        if let OverlapVerdict::Ok { end_to_end_ms, .. } = v {
            // Serial would be 100, overlap should be ~52.5.
            assert!(end_to_end_ms < 100.0);
        }
    }

    #[test]
    fn balanced_high_utilization() {
        // Balanced workloads (~equal compute/allreduce) overlap best
        // → low utilization_pct (lots of savings vs. serial baseline).
        let v_balanced = pick(50.0, 50.0);
        let v_unbalanced = pick(10.0, 100.0);
        if let (
            OverlapVerdict::Ok {
                utilization_pct: b, ..
            },
            OverlapVerdict::Ok {
                utilization_pct: u, ..
            },
        ) = (v_balanced, v_unbalanced)
        {
            assert!(b < u);
        }
    }

    #[test]
    fn boundary_at_half_ratio_overlap() {
        let v = pick(50.0, 100.0); // ratio = 0.5.
        if let OverlapVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, OverlapTier::Overlap);
        }
    }

    #[test]
    fn boundary_at_2_ratio_overlap() {
        let v = pick(100.0, 50.0); // ratio = 2.
        if let OverlapVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, OverlapTier::Overlap);
        }
    }
}
