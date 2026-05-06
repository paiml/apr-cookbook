//! # WASM Module-Size Cold-Start Budget
//!
//! Browser cold-start cost ~ module_size + compile_ms. Budget tiers:
//!
//! - module ≤ 256 KiB:   FCP-friendly (sub-100ms compile on V8)
//! - 256 KiB - 2 MiB:    OK with streaming compile
//! - 2 MiB - 8 MiB:      requires loading screen
//! - > 8 MiB:            needs split-bundle + lazy-load
//!
//! This recipe builds the classifier + lazy-load recommendation.
//!
//! Demonstrates the **WASM.14** recipe for PMAT-139 (wasm round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: V8 WASM streaming-compilation perf docs.
//!
//! Run with: cargo run --example wasm_module_size_budget
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeTier {
    Tiny,
    Small,
    Medium,
    Large,
    Excessive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStrategy {
    InlineFetch,
    StreamingCompile,
    LoadingScreen,
    SplitAndLazyLoad,
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok {
        tier: SizeTier,
        strategy: LoadStrategy,
        estimated_compile_ms: u32,
    },
    InvalidSize,
}

const ROUGH_KIB_PER_MS: u64 = 8;

pub fn classify(module_bytes: u64) -> BudgetVerdict {
    if module_bytes == 0 {
        return BudgetVerdict::InvalidSize;
    }
    let kib = module_bytes / 1024;
    let (tier, strategy) = if module_bytes <= 256 * 1024 {
        (SizeTier::Tiny, LoadStrategy::InlineFetch)
    } else if module_bytes <= 2 * 1024 * 1024 {
        (SizeTier::Small, LoadStrategy::StreamingCompile)
    } else if module_bytes <= 8 * 1024 * 1024 {
        (SizeTier::Medium, LoadStrategy::LoadingScreen)
    } else if module_bytes <= 64 * 1024 * 1024 {
        (SizeTier::Large, LoadStrategy::SplitAndLazyLoad)
    } else {
        (SizeTier::Excessive, LoadStrategy::SplitAndLazyLoad)
    };
    let estimated_compile_ms = (kib / ROUGH_KIB_PER_MS).max(1) as u32;
    BudgetVerdict::Ok {
        tier,
        strategy,
        estimated_compile_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_module_size_budget")?;

    let cases = [
        128 * 1024u64,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
    ];
    for size in cases {
        println!("{} bytes: {:?}", size, classify(size));
    }
    println!("zero: {:?}", classify(0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn tiny_module_inline_fetch() {
        let v = classify(128 * 1024);
        if let BudgetVerdict::Ok { tier, strategy, .. } = v {
            assert_eq!(tier, SizeTier::Tiny);
            assert_eq!(strategy, LoadStrategy::InlineFetch);
        }
    }

    #[test]
    fn small_module_streaming_compile() {
        let v = classify(1024 * 1024);
        if let BudgetVerdict::Ok { tier, strategy, .. } = v {
            assert_eq!(tier, SizeTier::Small);
            assert_eq!(strategy, LoadStrategy::StreamingCompile);
        }
    }

    #[test]
    fn medium_module_loading_screen() {
        let v = classify(4 * 1024 * 1024);
        if let BudgetVerdict::Ok { tier, strategy, .. } = v {
            assert_eq!(tier, SizeTier::Medium);
            assert_eq!(strategy, LoadStrategy::LoadingScreen);
        }
    }

    #[test]
    fn large_module_split_lazy() {
        let v = classify(16 * 1024 * 1024);
        if let BudgetVerdict::Ok { tier, strategy, .. } = v {
            assert_eq!(tier, SizeTier::Large);
            assert_eq!(strategy, LoadStrategy::SplitAndLazyLoad);
        }
    }

    #[test]
    fn excessive_module_split_lazy() {
        let v = classify(128 * 1024 * 1024);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SizeTier::Excessive);
        }
    }

    #[test]
    fn zero_bytes_invalid() {
        assert_eq!(classify(0), BudgetVerdict::InvalidSize);
    }

    #[test]
    fn estimated_compile_grows_with_size() {
        let v_small = classify(128 * 1024);
        let v_large = classify(16 * 1024 * 1024);
        if let (
            BudgetVerdict::Ok {
                estimated_compile_ms: s,
                ..
            },
            BudgetVerdict::Ok {
                estimated_compile_ms: l,
                ..
            },
        ) = (v_small, v_large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn boundary_at_256_kib_tiny() {
        let v = classify(256 * 1024);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SizeTier::Tiny);
        }
    }

    #[test]
    fn boundary_just_above_256_kib_small() {
        let v = classify(256 * 1024 + 1);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SizeTier::Small);
        }
    }

    #[test]
    fn estimated_compile_at_least_one_ms() {
        // Even tiniest module should report ≥ 1 ms.
        let v = classify(1);
        if let BudgetVerdict::Ok {
            estimated_compile_ms,
            ..
        } = v
        {
            assert!(estimated_compile_ms >= 1);
        }
    }
}
