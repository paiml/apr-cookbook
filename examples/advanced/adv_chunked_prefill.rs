//! # Advanced Chunked Prefill Window Sizer
//!
//! Long-context prefill blows a single GPU memory budget. Solution:
//! chunked prefill — process N tokens at a time, materializing only
//! that chunk's KV pieces.
//!
//! Window size = min(max_window, ceil(memory_budget_kib /
//! kv_per_token_kib)). Shorter windows = more passes but lower memory.
//!
//! Demonstrates the **ADV.10** recipe for PMAT-141 (advanced round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vLLM chunked-prefill optimization.
//!
//! Run with: cargo run --example adv_chunked_prefill
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ABS_MAX_WINDOW: u32 = 8_192;

#[derive(Debug, PartialEq)]
pub enum ChunkVerdict {
    Ok {
        window_tokens: u32,
        n_chunks: u32,
        peak_mem_kib: u64,
    },
    InvalidContext,
    InvalidMemoryBudget,
    InsufficientMemory,
}

pub fn plan(
    total_context_tokens: u32,
    kv_per_token_kib: u32,
    memory_budget_kib: u64,
) -> ChunkVerdict {
    if total_context_tokens == 0 {
        return ChunkVerdict::InvalidContext;
    }
    if memory_budget_kib == 0 {
        return ChunkVerdict::InvalidMemoryBudget;
    }
    if kv_per_token_kib == 0 {
        return ChunkVerdict::InvalidMemoryBudget;
    }
    let max_in_budget = memory_budget_kib / u64::from(kv_per_token_kib);
    if max_in_budget == 0 {
        return ChunkVerdict::InsufficientMemory;
    }
    let window = (max_in_budget as u32)
        .min(ABS_MAX_WINDOW)
        .min(total_context_tokens);
    let n_chunks = total_context_tokens.div_ceil(window);
    let peak_mem_kib = u64::from(window) * u64::from(kv_per_token_kib);
    ChunkVerdict::Ok {
        window_tokens: window,
        n_chunks,
        peak_mem_kib,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_chunked_prefill")?;

    println!(
        "32k context, 16 GiB budget: {:?}",
        plan(32_768, 64, 16 * 1024 * 1024)
    );
    println!(
        "100k context, 8 GiB budget: {:?}",
        plan(100_000, 64, 8 * 1024 * 1024)
    );
    println!("128 ctx with tiny budget: {:?}", plan(128, 64, 1024));
    println!("zero context: {:?}", plan(0, 64, 1024));
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
    fn fits_in_one_window() {
        // 1024 ctx, 64 KiB/token, 1 GiB budget → 16384 tokens fit but capped at 1024.
        let v = plan(1024, 64, 1 << 30);
        if let ChunkVerdict::Ok { n_chunks, .. } = v {
            assert_eq!(n_chunks, 1);
        }
    }

    #[test]
    fn many_chunks_for_long_context() {
        // 100k ctx, 64 KiB/token, 8 GiB budget = 8M / 64 = 131072 / abs max 8192 → window 8192.
        let v = plan(100_000, 64, 8 * 1024 * 1024);
        if let ChunkVerdict::Ok {
            window_tokens,
            n_chunks,
            ..
        } = v
        {
            assert!(window_tokens <= ABS_MAX_WINDOW);
            assert_eq!(n_chunks, 100_000_u32.div_ceil(window_tokens));
        }
    }

    #[test]
    fn invalid_context_rejected() {
        assert_eq!(plan(0, 64, 1024), ChunkVerdict::InvalidContext);
    }

    #[test]
    fn invalid_budget_rejected() {
        assert_eq!(plan(100, 64, 0), ChunkVerdict::InvalidMemoryBudget);
    }

    #[test]
    fn invalid_kv_per_token_rejected() {
        assert_eq!(plan(100, 0, 1024), ChunkVerdict::InvalidMemoryBudget);
    }

    #[test]
    fn insufficient_memory_rejected() {
        // Single token needs 64 KiB but budget is 32 KiB.
        let v = plan(100, 64, 32);
        assert_eq!(v, ChunkVerdict::InsufficientMemory);
    }

    #[test]
    fn peak_memory_within_budget() {
        let v = plan(100_000, 64, 8 * 1024 * 1024);
        if let ChunkVerdict::Ok { peak_mem_kib, .. } = v {
            assert!(peak_mem_kib <= 8 * 1024 * 1024);
        }
    }

    #[test]
    fn larger_budget_larger_window() {
        let v_small = plan(100_000, 64, 1024 * 1024);
        let v_large = plan(100_000, 64, 8 * 1024 * 1024);
        if let (
            ChunkVerdict::Ok {
                window_tokens: ws, ..
            },
            ChunkVerdict::Ok {
                window_tokens: wl, ..
            },
        ) = (v_small, v_large)
        {
            assert!(wl >= ws);
        }
    }

    #[test]
    fn window_capped_by_abs_max() {
        // Tons of memory; check we don't exceed ABS_MAX_WINDOW.
        let v = plan(1_000_000, 1, 100_000_000);
        if let ChunkVerdict::Ok { window_tokens, .. } = v {
            assert!(window_tokens <= ABS_MAX_WINDOW);
        }
    }

    #[test]
    fn n_chunks_covers_full_context() {
        let total_tokens = 100_000;
        let v = plan(total_tokens, 64, 8 * 1024 * 1024);
        if let ChunkVerdict::Ok {
            window_tokens,
            n_chunks,
            ..
        } = v
        {
            assert!(n_chunks * window_tokens >= total_tokens);
        }
    }

    #[test]
    fn small_context_one_chunk() {
        let v = plan(50, 64, 1 << 30);
        if let ChunkVerdict::Ok {
            window_tokens,
            n_chunks,
            ..
        } = v
        {
            assert_eq!(window_tokens, 50);
            assert_eq!(n_chunks, 1);
        }
    }
}
