//! # Advanced Long-Context Retrieval Split Picker
//!
//! Decide: stuff into context window directly, or chunk + RAG?
//!   total_tokens ≤ context_limit × 0.8 → InContext (no chunking)
//!   total_tokens ≤ context_limit × 4 → HybridChunkAndContext (top-K via RAG)
//!   total_tokens > context_limit × 4 → PureRagSearch (small context, big retrieval)
//!
//! Plus chunk-size + top-K recommendation.
//!
//! Demonstrates the **ADV.18** recipe for PMAT-149 (advanced round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: LangChain RetrievalQA + Anthropic long-context heuristics.
//!
//! Run with: cargo run --example adv_long_context_retrieval_split
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStrategy {
    InContext,
    HybridChunkAndContext,
    PureRagSearch,
}

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        strategy: SplitStrategy,
        chunk_size_tokens: u32,
        top_k_chunks: u32,
    },
    InvalidContext,
    InvalidTotalTokens,
}

pub fn pick(total_tokens: u32, context_limit_tokens: u32) -> SplitVerdict {
    if context_limit_tokens == 0 {
        return SplitVerdict::InvalidContext;
    }
    if total_tokens == 0 {
        return SplitVerdict::InvalidTotalTokens;
    }
    let context_80pct = (u64::from(context_limit_tokens) * 80) / 100;
    let context_4x = u64::from(context_limit_tokens) * 4;
    let total = u64::from(total_tokens);
    let strategy = if total <= context_80pct {
        SplitStrategy::InContext
    } else if total <= context_4x {
        SplitStrategy::HybridChunkAndContext
    } else {
        SplitStrategy::PureRagSearch
    };
    let chunk_size_tokens = match strategy {
        SplitStrategy::InContext => total_tokens,
        SplitStrategy::HybridChunkAndContext => 512,
        SplitStrategy::PureRagSearch => 256,
    };
    let top_k_chunks = match strategy {
        SplitStrategy::InContext => 1,
        SplitStrategy::HybridChunkAndContext => (context_limit_tokens / chunk_size_tokens).max(1),
        SplitStrategy::PureRagSearch => (context_limit_tokens / chunk_size_tokens / 2).max(1),
    };
    SplitVerdict::Ok {
        strategy,
        chunk_size_tokens,
        top_k_chunks,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_long_context_retrieval_split")?;

    println!("100 tokens / 4k ctx: {:?}", pick(100, 4096));
    println!("3000 tokens / 4k ctx: {:?}", pick(3000, 4096));
    println!("10000 tokens / 4k ctx: {:?}", pick(10_000, 4096));
    println!("100000 tokens / 4k ctx: {:?}", pick(100_000, 4096));
    println!("invalid: {:?}", pick(0, 4096));
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
    fn small_total_in_context() {
        let v = pick(100, 4096);
        if let SplitVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SplitStrategy::InContext);
        }
    }

    #[test]
    fn medium_total_hybrid() {
        let v = pick(10_000, 4096);
        if let SplitVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SplitStrategy::HybridChunkAndContext);
        }
    }

    #[test]
    fn large_total_pure_rag() {
        let v = pick(100_000, 4096);
        if let SplitVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SplitStrategy::PureRagSearch);
        }
    }

    #[test]
    fn invalid_zero_context() {
        assert_eq!(pick(100, 0), SplitVerdict::InvalidContext);
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(pick(0, 4096), SplitVerdict::InvalidTotalTokens);
    }

    #[test]
    fn in_context_uses_full_input() {
        let v = pick(100, 4096);
        if let SplitVerdict::Ok {
            chunk_size_tokens, ..
        } = v
        {
            assert_eq!(chunk_size_tokens, 100);
        }
    }

    #[test]
    fn hybrid_uses_512_chunks() {
        let v = pick(10_000, 4096);
        if let SplitVerdict::Ok {
            chunk_size_tokens, ..
        } = v
        {
            assert_eq!(chunk_size_tokens, 512);
        }
    }

    #[test]
    fn pure_rag_uses_smaller_chunks() {
        let v_hybrid = pick(10_000, 4096);
        let v_rag = pick(100_000, 4096);
        if let (
            SplitVerdict::Ok {
                chunk_size_tokens: hyb,
                ..
            },
            SplitVerdict::Ok {
                chunk_size_tokens: rag,
                ..
            },
        ) = (v_hybrid, v_rag)
        {
            assert!(rag < hyb);
        }
    }

    #[test]
    fn boundary_at_80pct_in_context() {
        let v = pick(3276, 4096); // 80% of 4096 = 3276.8.
        if let SplitVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SplitStrategy::InContext);
        }
    }

    #[test]
    fn just_above_80pct_hybrid() {
        let v = pick(3300, 4096);
        if let SplitVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, SplitStrategy::HybridChunkAndContext);
        }
    }

    #[test]
    fn top_k_at_least_one() {
        for total in [100, 10_000, 100_000] {
            if let SplitVerdict::Ok { top_k_chunks, .. } = pick(total, 4096) {
                assert!(top_k_chunks >= 1);
            }
        }
    }
}
