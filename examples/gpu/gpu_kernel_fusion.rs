//! # GPU Kernel-Fusion Planner
//!
//! Fuse epilogue ops (bias-add, activation, dropout) into matmul/conv
//! kernel to save:
//!   global memory bandwidth (intermediate tensor not materialized)
//!   kernel-launch overhead
//!
//! Picker rules:
//!   matmul + bias + gelu → ALL fused (cuBLASLt epilogue)
//!   matmul + softmax → too complex, skip
//!   matmul + sum-reduce → fuse partial (k-major reduction)
//!
//! Demonstrates the **GPU.31** recipe for PMAT-146 (gpu round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA cuBLASLt documentation § epilogue ops.
//!
//! Run with: cargo run --example gpu_kernel_fusion
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpilogueOp {
    None,
    BiasAdd,
    Gelu,
    Relu,
    SiLU,
    Softmax,
    SumReduce,
    Dropout,
}

#[derive(Debug, PartialEq)]
pub enum FusionVerdict {
    FullyFused {
        ops: Vec<EpilogueOp>,
        bandwidth_savings_pct: u32,
    },
    PartiallyFused {
        fused: Vec<EpilogueOp>,
        skipped: Vec<EpilogueOp>,
    },
    NoFusion {
        reason: &'static str,
    },
    EmptyOps,
}

pub fn plan(epilogue_ops: &[EpilogueOp]) -> FusionVerdict {
    if epilogue_ops.is_empty() {
        return FusionVerdict::EmptyOps;
    }
    let mut fused: Vec<EpilogueOp> = Vec::new();
    let mut skipped: Vec<EpilogueOp> = Vec::new();
    for &op in epilogue_ops {
        match op {
            EpilogueOp::None => {}
            EpilogueOp::BiasAdd
            | EpilogueOp::Gelu
            | EpilogueOp::Relu
            | EpilogueOp::SiLU
            | EpilogueOp::Dropout => fused.push(op),
            EpilogueOp::Softmax => skipped.push(op),
            EpilogueOp::SumReduce => fused.push(op),
        }
    }
    if fused.is_empty() && skipped.is_empty() {
        return FusionVerdict::NoFusion {
            reason: "no fusable ops",
        };
    }
    if skipped.is_empty() {
        let bandwidth_savings_pct = (fused.len() as u32 * 25).min(80);
        return FusionVerdict::FullyFused {
            ops: fused,
            bandwidth_savings_pct,
        };
    }
    FusionVerdict::PartiallyFused { fused, skipped }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_kernel_fusion")?;

    println!(
        "matmul + bias + gelu: {:?}",
        plan(&[EpilogueOp::BiasAdd, EpilogueOp::Gelu])
    );
    println!(
        "matmul + bias + softmax: {:?}",
        plan(&[EpilogueOp::BiasAdd, EpilogueOp::Softmax])
    );
    println!("only None: {:?}", plan(&[EpilogueOp::None]));
    println!("empty: {:?}", plan(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn bias_gelu_fully_fused() {
        let v = plan(&[EpilogueOp::BiasAdd, EpilogueOp::Gelu]);
        assert!(matches!(v, FusionVerdict::FullyFused { .. }));
    }

    #[test]
    fn softmax_skipped() {
        let v = plan(&[EpilogueOp::BiasAdd, EpilogueOp::Softmax]);
        if let FusionVerdict::PartiallyFused { fused, skipped } = v {
            assert_eq!(fused, vec![EpilogueOp::BiasAdd]);
            assert_eq!(skipped, vec![EpilogueOp::Softmax]);
        }
    }

    #[test]
    fn empty_ops_rejected() {
        assert_eq!(plan(&[]), FusionVerdict::EmptyOps);
    }

    #[test]
    fn only_none_no_fusion() {
        let v = plan(&[EpilogueOp::None]);
        assert!(matches!(v, FusionVerdict::NoFusion { .. }));
    }

    #[test]
    fn relu_fusable() {
        let v = plan(&[EpilogueOp::Relu]);
        assert!(matches!(v, FusionVerdict::FullyFused { .. }));
    }

    #[test]
    fn silu_fusable() {
        let v = plan(&[EpilogueOp::SiLU]);
        assert!(matches!(v, FusionVerdict::FullyFused { .. }));
    }

    #[test]
    fn dropout_fusable() {
        let v = plan(&[EpilogueOp::Dropout]);
        assert!(matches!(v, FusionVerdict::FullyFused { .. }));
    }

    #[test]
    fn sum_reduce_fusable() {
        let v = plan(&[EpilogueOp::SumReduce]);
        assert!(matches!(v, FusionVerdict::FullyFused { .. }));
    }

    #[test]
    fn savings_increase_with_fused_count() {
        let v_one = plan(&[EpilogueOp::BiasAdd]);
        let v_two = plan(&[EpilogueOp::BiasAdd, EpilogueOp::Gelu]);
        if let (
            FusionVerdict::FullyFused {
                bandwidth_savings_pct: a,
                ..
            },
            FusionVerdict::FullyFused {
                bandwidth_savings_pct: b,
                ..
            },
        ) = (v_one, v_two)
        {
            assert!(b > a);
        }
    }

    #[test]
    fn savings_capped_at_80() {
        let many = vec![
            EpilogueOp::BiasAdd,
            EpilogueOp::Gelu,
            EpilogueOp::Dropout,
            EpilogueOp::SumReduce,
            EpilogueOp::Relu,
        ];
        let v = plan(&many);
        if let FusionVerdict::FullyFused {
            bandwidth_savings_pct,
            ..
        } = v
        {
            assert!(bandwidth_savings_pct <= 80);
        }
    }

    #[test]
    fn mixed_fusable_and_skipped() {
        let v = plan(&[EpilogueOp::BiasAdd, EpilogueOp::Gelu, EpilogueOp::Softmax]);
        if let FusionVerdict::PartiallyFused { fused, skipped } = v {
            assert_eq!(fused.len(), 2);
            assert_eq!(skipped.len(), 1);
        }
    }
}
