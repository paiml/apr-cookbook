//! # apr serve --kv-cache-bytes — KV Cache Memory Budget
//!
//! KV cache memory grows linearly with context: bytes = 2 (K+V) ×
//! n_layers × n_heads × head_dim × seq_len × dtype_bytes × batch.
//! This recipe builds the calculator + budget gate against a max.
//!
//! Demonstrates the **SERVE.6** recipe for PMAT-116 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-001 + Pope et al. 2022 (KV cache scaling)
//!
//! Run with: cargo run --example cli_serve_kv_cache_budget
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok { bytes: u64 },
    ExceedsBudget { bytes: u64, budget: u64 },
    InvalidShape,
}

#[derive(Debug, Clone, Copy)]
pub struct KvShape {
    pub n_layers: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub batch: u32,
    pub dtype_bytes: u32,
}

pub fn estimate_bytes(shape: KvShape) -> Option<u64> {
    if shape.n_layers == 0
        || shape.n_heads == 0
        || shape.head_dim == 0
        || shape.seq_len == 0
        || shape.batch == 0
        || shape.dtype_bytes == 0
    {
        return None;
    }
    let mut acc = 2u64; // K + V
    for v in [
        shape.n_layers,
        shape.n_heads,
        shape.head_dim,
        shape.seq_len,
        shape.batch,
        shape.dtype_bytes,
    ] {
        acc = acc.checked_mul(u64::from(v))?;
    }
    Some(acc)
}

pub fn validate(shape: KvShape, budget_bytes: u64) -> BudgetVerdict {
    let Some(bytes) = estimate_bytes(shape) else {
        return BudgetVerdict::InvalidShape;
    };
    if bytes > budget_bytes {
        BudgetVerdict::ExceedsBudget {
            bytes,
            budget: budget_bytes,
        }
    } else {
        BudgetVerdict::Ok { bytes }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_kv_cache_budget")?;

    // 7B-style: 32 layers, 32 heads, 128 head_dim, FP16 (2 bytes).
    let shape = KvShape {
        n_layers: 32,
        n_heads: 32,
        head_dim: 128,
        seq_len: 4096,
        batch: 1,
        dtype_bytes: 2,
    };
    let budget = 4 * 1024 * 1024 * 1024; // 4 GiB
    println!("7B @ 4K seq: {:?}", validate(shape, budget));

    let big_shape = KvShape {
        seq_len: 32_000,
        ..shape
    };
    println!("7B @ 32K seq: {:?}", validate(big_shape, budget));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> KvShape {
        KvShape {
            n_layers: 4,
            n_heads: 4,
            head_dim: 64,
            seq_len: 256,
            batch: 1,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_shape_estimates_correctly() {
        // 2 × 4 × 4 × 64 × 256 × 1 × 2 = 1,048,576 bytes
        assert_eq!(estimate_bytes(small()), Some(1_048_576));
    }

    #[test]
    fn zero_dimension_yields_none() {
        let mut s = small();
        s.n_heads = 0;
        assert!(estimate_bytes(s).is_none());
        s = small();
        s.dtype_bytes = 0;
        assert!(estimate_bytes(s).is_none());
    }

    #[test]
    fn validate_within_budget_passes() {
        let v = validate(small(), 10_000_000);
        assert!(matches!(v, BudgetVerdict::Ok { bytes: 1_048_576 }));
    }

    #[test]
    fn validate_exceeds_budget_rejected() {
        let v = validate(small(), 100);
        assert!(matches!(v, BudgetVerdict::ExceedsBudget { .. }));
    }

    #[test]
    fn validate_invalid_shape_returns_invalid_shape() {
        let mut s = small();
        s.batch = 0;
        assert_eq!(validate(s, 1_000_000), BudgetVerdict::InvalidShape);
    }

    #[test]
    fn fp32_doubles_bytes_vs_fp16() {
        let mut fp16 = small();
        fp16.dtype_bytes = 2;
        let mut fp32 = small();
        fp32.dtype_bytes = 4;
        assert_eq!(
            estimate_bytes(fp32).unwrap(),
            estimate_bytes(fp16).unwrap() * 2
        );
    }

    #[test]
    fn batch_doubles_bytes() {
        let one = estimate_bytes(small()).unwrap();
        let two = estimate_bytes(KvShape {
            batch: 2,
            ..small()
        })
        .unwrap();
        assert_eq!(two, one * 2);
    }

    #[test]
    fn seq_len_doubles_bytes() {
        let s1 = estimate_bytes(small()).unwrap();
        let s2 = estimate_bytes(KvShape {
            seq_len: 512,
            ..small()
        })
        .unwrap();
        assert_eq!(s2, s1 * 2);
    }

    #[test]
    fn shape_7b_at_4k_seq_equals_2gib() {
        // Sanity: 7B-class @ 4K seq, FP16, batch 1 ≈ 2 GiB.
        let s = KvShape {
            n_layers: 32,
            n_heads: 32,
            head_dim: 128,
            seq_len: 4096,
            batch: 1,
            dtype_bytes: 2,
        };
        let bytes = estimate_bytes(s).unwrap();
        // 2 × 32 × 32 × 128 × 4096 × 1 × 2 = 2,147,483,648 (= 2 GiB).
        assert_eq!(bytes, 2 * 1024 * 1024 * 1024);
    }
}
