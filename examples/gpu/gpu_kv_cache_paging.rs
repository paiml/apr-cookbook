//! # GPU KV-Cache Paging Planner
//!
//! Page-attention (vLLM): KV cache split into fixed-size blocks
//! (16/32/64 tokens). Memory budget = total_kv_bytes / page_bytes →
//! number of pages. Per-token KV bytes = 2 × layers × heads × head_dim
//! × dtype_bytes (× 2 for K + V). This recipe builds the planner.
//!
//! Demonstrates the **GPU.20** recipe for PMAT-137 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. arXiv:2309.06180.
//!
//! Run with: cargo run --example gpu_kv_cache_paging
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PageVerdict {
    Ok {
        bytes_per_token: u64,
        bytes_per_page: u64,
        total_pages: u64,
    },
    InvalidShape,
    InvalidBudget,
    InvalidPageSize,
}

pub fn plan(
    layers: u32,
    heads: u32,
    head_dim: u32,
    dtype_bytes: u32,
    page_size_tokens: u32,
    total_kv_budget_bytes: u64,
) -> PageVerdict {
    if layers == 0 || heads == 0 || head_dim == 0 || dtype_bytes == 0 {
        return PageVerdict::InvalidShape;
    }
    if total_kv_budget_bytes == 0 {
        return PageVerdict::InvalidBudget;
    }
    if page_size_tokens == 0 {
        return PageVerdict::InvalidPageSize;
    }
    // K + V each take 1 × layers × heads × head_dim × dtype_bytes per token.
    let bytes_per_token =
        2u64 * u64::from(layers) * u64::from(heads) * u64::from(head_dim) * u64::from(dtype_bytes);
    let bytes_per_page = bytes_per_token * u64::from(page_size_tokens);
    let total_pages = total_kv_budget_bytes / bytes_per_page;
    PageVerdict::Ok {
        bytes_per_token,
        bytes_per_page,
        total_pages,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_kv_cache_paging")?;

    // Llama-7B-ish: 32 layers × 32 heads × 128 head_dim × 2 bytes (fp16).
    println!(
        "llama-7b 16-tok pages, 8 GiB: {:?}",
        plan(32, 32, 128, 2, 16, 8 * 1024 * 1024 * 1024)
    );

    println!(
        "llama-70b 32-tok pages, 24 GiB: {:?}",
        plan(80, 64, 128, 2, 32, 24 * 1024 * 1024 * 1024)
    );

    println!("invalid layers=0: {:?}", plan(0, 32, 128, 2, 16, 1 << 30));
    println!("invalid budget=0: {:?}", plan(32, 32, 128, 2, 16, 0));
    println!("invalid page=0: {:?}", plan(32, 32, 128, 2, 0, 1 << 30));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paging_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_llama_7b_planner() {
        // 2 × 32L × 32H × 128D × 2bytes = 524288 bytes/token.
        let v = plan(32, 32, 128, 2, 16, 8 * 1024 * 1024 * 1024);
        if let PageVerdict::Ok {
            bytes_per_token, ..
        } = v
        {
            assert_eq!(bytes_per_token, 524_288);
        }
    }

    #[test]
    fn page_size_scales_bytes_per_page() {
        // 16-token vs 32-token pages → 32-token has 2× bytes per page.
        let v16 = plan(32, 32, 128, 2, 16, 1 << 30);
        let v32 = plan(32, 32, 128, 2, 32, 1 << 30);
        if let (
            PageVerdict::Ok {
                bytes_per_page: b16,
                ..
            },
            PageVerdict::Ok {
                bytes_per_page: b32,
                ..
            },
        ) = (v16, v32)
        {
            assert_eq!(b32, b16 * 2);
        }
    }

    #[test]
    fn larger_budget_more_pages() {
        let small = plan(32, 32, 128, 2, 16, 1u64 << 30);
        let large = plan(32, 32, 128, 2, 16, 8u64 << 30);
        if let (
            PageVerdict::Ok {
                total_pages: p_small,
                ..
            },
            PageVerdict::Ok {
                total_pages: p_large,
                ..
            },
        ) = (small, large)
        {
            assert_eq!(p_large, p_small * 8);
        }
    }

    #[test]
    fn fp32_doubles_bytes_per_token_vs_fp16() {
        let fp16 = plan(32, 32, 128, 2, 16, 1 << 30);
        let fp32 = plan(32, 32, 128, 4, 16, 1 << 30);
        if let (
            PageVerdict::Ok {
                bytes_per_token: b16,
                ..
            },
            PageVerdict::Ok {
                bytes_per_token: b32,
                ..
            },
        ) = (fp16, fp32)
        {
            assert_eq!(b32, b16 * 2);
        }
    }

    #[test]
    fn zero_layers_invalid() {
        assert_eq!(plan(0, 32, 128, 2, 16, 1 << 30), PageVerdict::InvalidShape);
    }

    #[test]
    fn zero_heads_invalid() {
        assert_eq!(plan(32, 0, 128, 2, 16, 1 << 30), PageVerdict::InvalidShape);
    }

    #[test]
    fn zero_head_dim_invalid() {
        assert_eq!(plan(32, 32, 0, 2, 16, 1 << 30), PageVerdict::InvalidShape);
    }

    #[test]
    fn zero_budget_invalid() {
        assert_eq!(plan(32, 32, 128, 2, 16, 0), PageVerdict::InvalidBudget);
    }

    #[test]
    fn zero_page_size_invalid() {
        assert_eq!(
            plan(32, 32, 128, 2, 0, 1 << 30),
            PageVerdict::InvalidPageSize
        );
    }

    #[test]
    fn k_and_v_factor_is_two() {
        // 1L × 1H × 1D × 1byte × 1tok page → bytes_per_token = 2 (K + V).
        let v = plan(1, 1, 1, 1, 1, 100);
        if let PageVerdict::Ok {
            bytes_per_token, ..
        } = v
        {
            assert_eq!(bytes_per_token, 2);
        }
    }
}
