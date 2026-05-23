//! # GPU Tensor-Parallel Split Planner
//!
//! Tensor parallelism (TP) splits a layer's weight tensor across N GPUs.
//! For attention: split heads across GPUs (need heads % N == 0).
//! For FFN: split intermediate dim (need intermediate % N == 0 ideally).
//! For embedding: split vocab dim. This recipe picks the split + flags
//! when degree is unevenly divisible.
//!
//! Demonstrates the **GPU.21** recipe for PMAT-137 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Megatron-LM tensor parallelism (Shoeybi et al. 2019).
//!
//! Run with: cargo run --example gpu_tensor_parallel_split
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    Ffn,
    Embedding,
}

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        split_dim: usize,
        per_gpu_size: u32,
    },
    UnevenSplit {
        dim: u32,
        tp_degree: u32,
        remainder: u32,
    },
    InvalidDegree,
    InvalidDim,
}

pub fn plan(layer: LayerKind, heads_or_dim: u32, tp_degree: u32) -> SplitVerdict {
    if tp_degree == 0 {
        return SplitVerdict::InvalidDegree;
    }
    if heads_or_dim == 0 {
        return SplitVerdict::InvalidDim;
    }
    let split_dim = match layer {
        LayerKind::Attention => 0,
        LayerKind::Ffn => 1,
        LayerKind::Embedding => 0,
    };
    let remainder = heads_or_dim % tp_degree;
    if remainder != 0 {
        return SplitVerdict::UnevenSplit {
            dim: heads_or_dim,
            tp_degree,
            remainder,
        };
    }
    SplitVerdict::Ok {
        split_dim,
        per_gpu_size: heads_or_dim / tp_degree,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_tensor_parallel_split")?;

    println!(
        "attn 32 heads / 8 GPU: {:?}",
        plan(LayerKind::Attention, 32, 8)
    );
    println!(
        "ffn 11008 dim / 8 GPU: {:?}",
        plan(LayerKind::Ffn, 11008, 8)
    );
    println!(
        "attn 32 heads / 5 GPU: {:?}",
        plan(LayerKind::Attention, 32, 5)
    );
    println!("invalid: {:?}", plan(LayerKind::Attention, 0, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn attention_even_split() {
        let v = plan(LayerKind::Attention, 32, 8);
        if let SplitVerdict::Ok { per_gpu_size, .. } = v {
            assert_eq!(per_gpu_size, 4);
        }
    }

    #[test]
    fn ffn_even_split() {
        let v = plan(LayerKind::Ffn, 11008, 8);
        if let SplitVerdict::Ok { per_gpu_size, .. } = v {
            assert_eq!(per_gpu_size, 1376);
        }
    }

    #[test]
    fn embedding_even_split() {
        // 32000 vocab / 4 GPU = 8000.
        let v = plan(LayerKind::Embedding, 32_000, 4);
        if let SplitVerdict::Ok { per_gpu_size, .. } = v {
            assert_eq!(per_gpu_size, 8000);
        }
    }

    #[test]
    fn uneven_split_flagged() {
        // 32 heads / 5 GPUs = remainder 2.
        let v = plan(LayerKind::Attention, 32, 5);
        assert!(matches!(v, SplitVerdict::UnevenSplit { remainder: 2, .. }));
    }

    #[test]
    fn zero_degree_invalid() {
        assert_eq!(
            plan(LayerKind::Attention, 32, 0),
            SplitVerdict::InvalidDegree
        );
    }

    #[test]
    fn zero_dim_invalid() {
        assert_eq!(plan(LayerKind::Attention, 0, 8), SplitVerdict::InvalidDim);
    }

    #[test]
    fn attention_split_dim_is_zero() {
        if let SplitVerdict::Ok { split_dim, .. } = plan(LayerKind::Attention, 32, 8) {
            assert_eq!(split_dim, 0);
        }
    }

    #[test]
    fn ffn_split_dim_is_one() {
        if let SplitVerdict::Ok { split_dim, .. } = plan(LayerKind::Ffn, 11008, 8) {
            assert_eq!(split_dim, 1);
        }
    }

    #[test]
    fn single_gpu_full_size() {
        // tp_degree=1 → no split, per_gpu_size == full dim.
        let v = plan(LayerKind::Attention, 32, 1);
        if let SplitVerdict::Ok { per_gpu_size, .. } = v {
            assert_eq!(per_gpu_size, 32);
        }
    }

    #[test]
    fn degree_above_dim_uneven() {
        // 8 heads / 16 GPUs → can't even split.
        let v = plan(LayerKind::Attention, 8, 16);
        assert!(matches!(v, SplitVerdict::UnevenSplit { .. }));
    }
}
