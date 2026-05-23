//! # Training Long-Context RoPE Extension Picker
//!
//! Extending model context (e.g. 4k → 128k) requires scaling RoPE
//! base θ. Strategies:
//!   PiAware (linear position interpolation): cheap, slight quality loss
//!   NTK-aware: scale theta non-linearly; better quality
//!   YaRN: optimized NTK + truncation; best quality, more complex
//!
//! Picker: given (original_ctx, target_ctx, quality_priority), pick.
//!
//! Demonstrates the **TRAIN.18** recipe for PMAT-146 (training round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Peng et al. (2023). YaRN: Efficient Context Window Extension. arXiv:2309.00071.
//!
//! Run with: cargo run --example training_long_context_extension
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionStrategy {
    PiAware,
    NtkAware,
    YaRN,
}

#[derive(Debug, PartialEq)]
pub enum ExtensionVerdict {
    Ok {
        strategy: ExtensionStrategy,
        scale_factor: f64,
        recommended_finetune_tokens: u64,
    },
    NoExtensionNeeded,
    InvalidContext,
    UnreasonableExtension {
        ratio: u32,
    },
}

const MAX_REASONABLE_RATIO: u32 = 32;

pub fn pick(original_ctx: u32, target_ctx: u32, quality_priority: bool) -> ExtensionVerdict {
    if original_ctx == 0 || target_ctx == 0 {
        return ExtensionVerdict::InvalidContext;
    }
    if target_ctx <= original_ctx {
        return ExtensionVerdict::NoExtensionNeeded;
    }
    let ratio = target_ctx / original_ctx;
    if ratio > MAX_REASONABLE_RATIO {
        return ExtensionVerdict::UnreasonableExtension { ratio };
    }
    let strategy = if quality_priority {
        ExtensionStrategy::YaRN
    } else if ratio <= 2 {
        ExtensionStrategy::PiAware
    } else {
        ExtensionStrategy::NtkAware
    };
    let scale_factor = f64::from(target_ctx) / f64::from(original_ctx);
    let tokens_per_million_ctx = match strategy {
        ExtensionStrategy::PiAware => 1_000_000_u64,
        ExtensionStrategy::NtkAware => 5_000_000,
        ExtensionStrategy::YaRN => 10_000_000,
    };
    let recommended_finetune_tokens = tokens_per_million_ctx * (target_ctx as u64) / 4096;
    ExtensionVerdict::Ok {
        strategy,
        scale_factor,
        recommended_finetune_tokens,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_long_context_extension")?;

    println!("4k → 8k (cheap): {:?}", pick(4096, 8192, false));
    println!("4k → 32k (better): {:?}", pick(4096, 32_768, false));
    println!("4k → 128k (best): {:?}", pick(4096, 131_072, true));
    println!("no ext: {:?}", pick(4096, 4096, false));
    println!("absurd: {:?}", pick(4096, 1_000_000, false));
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
    fn small_ratio_picks_pi_aware() {
        let v = pick(4096, 8192, false);
        if let ExtensionVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, ExtensionStrategy::PiAware);
        }
    }

    #[test]
    fn medium_ratio_picks_ntk_aware() {
        let v = pick(4096, 32_768, false);
        if let ExtensionVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, ExtensionStrategy::NtkAware);
        }
    }

    #[test]
    fn quality_priority_picks_yarn() {
        let v = pick(4096, 16_384, true);
        if let ExtensionVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, ExtensionStrategy::YaRN);
        }
    }

    #[test]
    fn no_extension_needed() {
        assert_eq!(pick(4096, 4096, false), ExtensionVerdict::NoExtensionNeeded);
    }

    #[test]
    fn target_below_original_no_extension() {
        assert_eq!(pick(4096, 2048, false), ExtensionVerdict::NoExtensionNeeded);
    }

    #[test]
    fn unreasonable_ratio_rejected() {
        let v = pick(4096, 1_000_000, false);
        assert!(matches!(v, ExtensionVerdict::UnreasonableExtension { .. }));
    }

    #[test]
    fn invalid_zero_original() {
        assert_eq!(pick(0, 8192, false), ExtensionVerdict::InvalidContext);
    }

    #[test]
    fn invalid_zero_target() {
        assert_eq!(pick(4096, 0, false), ExtensionVerdict::InvalidContext);
    }

    #[test]
    fn scale_factor_correct() {
        let v = pick(4096, 16_384, false);
        if let ExtensionVerdict::Ok { scale_factor, .. } = v {
            assert!((scale_factor - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn yarn_more_tokens_than_pi() {
        let v_yarn = pick(4096, 16_384, true);
        let v_pi = pick(4096, 8192, false); // Pi-aware
        if let (
            ExtensionVerdict::Ok {
                recommended_finetune_tokens: yarn,
                ..
            },
            ExtensionVerdict::Ok {
                recommended_finetune_tokens: pi,
                ..
            },
        ) = (v_yarn, v_pi)
        {
            assert!(yarn > pi);
        }
    }

    #[test]
    fn ratio_2_uses_pi_aware() {
        // exactly 2× → PiAware (cheap path).
        let v = pick(4096, 8192, false);
        if let ExtensionVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, ExtensionStrategy::PiAware);
        }
    }
}
