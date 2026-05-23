//! # SIMD Horizontal Reduce Dispatcher
//!
//! Horizontal reductions (sum/min/max across SIMD lanes) have ISA-
//! specific intrinsics: AVX uses `_mm256_hadd_ps` + extract; AVX-512
//! has `_mm512_reduce_*`. Lane width determines tree depth: 4 lanes
//! → 2 levels, 8 → 3, 16 → 4. This recipe dispatches the operation
//! + reports tree depth.
//!
//! Demonstrates the **SIMD.7** recipe for PMAT-123 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel Intrinsics Guide §reduction.
//!
//! Run with: cargo run --example simd_horizontal_reduce_dispatcher
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok { intrinsic: String, tree_depth: u32 },
    UnsupportedLaneCount,
    InvalidEmptyVector,
}

pub fn dispatch(op: ReduceOp, lane_count: u32) -> DispatchVerdict {
    if lane_count == 0 {
        return DispatchVerdict::InvalidEmptyVector;
    }
    if !lane_count.is_power_of_two() {
        return DispatchVerdict::UnsupportedLaneCount;
    }
    if !(2..=16).contains(&lane_count) {
        return DispatchVerdict::UnsupportedLaneCount;
    }
    let intrinsic = match (op, lane_count) {
        (ReduceOp::Sum, 4) => "_mm_hadd_ps + extract".into(),
        (ReduceOp::Sum, 8) => "_mm256_hadd_ps + extract".into(),
        (ReduceOp::Sum, 16) => "_mm512_reduce_add_ps".into(),
        (ReduceOp::Min, n) => format!("_mm{}_reduce_min_ps", n * 32),
        (ReduceOp::Max, n) => format!("_mm{}_reduce_max_ps", n * 32),
        (ReduceOp::Sum, 2) => "_mm_add_ps + scalar".into(),
        _ => "scalar fallback".into(),
    };
    DispatchVerdict::Ok {
        intrinsic,
        tree_depth: lane_count.trailing_zeros(),
    }
}

pub fn reduce_scalar(op: ReduceOp, values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let result = match op {
        ReduceOp::Sum => values.iter().sum(),
        ReduceOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        ReduceOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };
    Some(result)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_horizontal_reduce_dispatcher")?;

    for op in [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max] {
        for lanes in [2u32, 4, 8, 16, 32, 3] {
            println!("{op:?} lanes={lanes}  →  {:?}", dispatch(op, lanes));
        }
    }
    let v = [1.0, 2.0, 3.0, 4.0];
    for op in [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max] {
        println!("scalar {op:?}: {:?}", reduce_scalar(op, &v));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn sum_8_lanes_uses_avx() {
        let v = dispatch(ReduceOp::Sum, 8);
        if let DispatchVerdict::Ok { intrinsic, .. } = v {
            assert!(intrinsic.contains("256"));
        }
    }

    #[test]
    fn sum_16_lanes_uses_avx512() {
        let v = dispatch(ReduceOp::Sum, 16);
        if let DispatchVerdict::Ok { intrinsic, .. } = v {
            assert!(intrinsic.contains("512"));
        }
    }

    #[test]
    fn tree_depth_log2_of_lanes() {
        if let DispatchVerdict::Ok { tree_depth, .. } = dispatch(ReduceOp::Sum, 4) {
            assert_eq!(tree_depth, 2);
        }
        if let DispatchVerdict::Ok { tree_depth, .. } = dispatch(ReduceOp::Sum, 16) {
            assert_eq!(tree_depth, 4);
        }
    }

    #[test]
    fn non_power_of_two_rejected() {
        assert_eq!(
            dispatch(ReduceOp::Sum, 3),
            DispatchVerdict::UnsupportedLaneCount
        );
        assert_eq!(
            dispatch(ReduceOp::Sum, 7),
            DispatchVerdict::UnsupportedLaneCount
        );
    }

    #[test]
    fn over_16_lanes_unsupported() {
        // Single-instruction reduce caps at 16 lanes (AVX-512).
        assert_eq!(
            dispatch(ReduceOp::Sum, 32),
            DispatchVerdict::UnsupportedLaneCount
        );
    }

    #[test]
    fn zero_lanes_invalid() {
        assert_eq!(
            dispatch(ReduceOp::Sum, 0),
            DispatchVerdict::InvalidEmptyVector
        );
    }

    #[test]
    fn scalar_sum_correct() {
        assert_eq!(reduce_scalar(ReduceOp::Sum, &[1.0, 2.0, 3.0]), Some(6.0));
    }

    #[test]
    fn scalar_min_max_correct() {
        let v = [3.0, 1.0, 4.0, 1.5];
        assert_eq!(reduce_scalar(ReduceOp::Min, &v), Some(1.0));
        assert_eq!(reduce_scalar(ReduceOp::Max, &v), Some(4.0));
    }

    #[test]
    fn scalar_empty_yields_none() {
        assert!(reduce_scalar(ReduceOp::Sum, &[]).is_none());
    }
}
