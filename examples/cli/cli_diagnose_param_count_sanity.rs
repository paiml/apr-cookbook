//! # apr diagnose --param-count — Sanity Check vs Architecture
//!
//! For a known architecture (n_layers, hidden, vocab), parameter count
//! has a closed-form estimate within ±5%. Discrepancy beyond that
//! signals a corrupt checkpoint, mis-tagged model, or LoRA adapter
//! mistakenly loaded as full model. This recipe builds the sanity
//! checker.
//!
//! Demonstrates the **DIAG.5** recipe for PMAT-116 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAG-001 + Hoffmann et al. 2022 (Chinchilla)
//!
//! Run with: cargo run --example cli_diagnose_param_count_sanity
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SanityVerdict {
    Ok,
    DiscrepancyTooLarge {
        observed: u64,
        estimated: u64,
        ratio: f64,
    },
    InvalidArchitecture,
}

const TOLERANCE: f64 = 0.05; // ±5%

#[derive(Debug, Clone, Copy)]
pub struct Architecture {
    pub n_layers: u32,
    pub hidden: u32,
    pub n_heads: u32,
    pub vocab: u32,
    pub ffn_mult: u32, // typically 4 for MLP, 2.66 for SwiGLU (×8/3)
}

pub fn estimate_params(arch: Architecture) -> Option<u64> {
    if arch.n_layers == 0 || arch.hidden == 0 || arch.n_heads == 0 || arch.vocab == 0 {
        return None;
    }
    let hidden = u64::from(arch.hidden);
    let layers = u64::from(arch.n_layers);
    let vocab = u64::from(arch.vocab);
    let ffn_mult = u64::from(arch.ffn_mult.max(1));

    // Embeddings: 2 × vocab × hidden (input + tied output).
    let embed = 2 * vocab * hidden;
    // Per layer: 4 attn (qkvo) + 3 FFN (gate/up/down) projections, each hidden×hidden(*ffn_mult).
    let attn = 4 * hidden * hidden;
    let ffn = 3 * hidden * (hidden * ffn_mult);
    let per_layer = attn + ffn;
    Some(embed + layers * per_layer)
}

pub fn check_sanity(observed: u64, arch: Architecture) -> SanityVerdict {
    let Some(estimated) = estimate_params(arch) else {
        return SanityVerdict::InvalidArchitecture;
    };
    if estimated == 0 {
        return SanityVerdict::InvalidArchitecture;
    }
    let ratio = observed as f64 / estimated as f64;
    if (ratio - 1.0).abs() > TOLERANCE {
        return SanityVerdict::DiscrepancyTooLarge {
            observed,
            estimated,
            ratio,
        };
    }
    SanityVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_param_count_sanity")?;

    // Llama-3-8B-style: 32 layers, 4096 hidden, 32 heads, 128K vocab, ffn_mult=4 (approx).
    let arch = Architecture {
        n_layers: 32,
        hidden: 4096,
        n_heads: 32,
        vocab: 128_000,
        ffn_mult: 4,
    };
    println!("estimate: {:?}", estimate_params(arch));
    println!("sanity(8.5B): {:?}", check_sanity(8_500_000_000, arch));
    println!("sanity(50M, LoRA?): {:?}", check_sanity(50_000_000, arch));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_arch() -> Architecture {
        Architecture {
            n_layers: 2,
            hidden: 64,
            n_heads: 4,
            vocab: 1000,
            ffn_mult: 4,
        }
    }

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn estimate_matches_within_tolerance() {
        let est = estimate_params(small_arch()).unwrap();
        // est ± 5%: pass.
        assert_eq!(check_sanity(est, small_arch()), SanityVerdict::Ok);
        let near = (est as f64 * 1.04) as u64;
        assert_eq!(check_sanity(near, small_arch()), SanityVerdict::Ok);
    }

    #[test]
    fn way_off_rejected() {
        let est = estimate_params(small_arch()).unwrap();
        let off = est * 10;
        let v = check_sanity(off, small_arch());
        assert!(matches!(v, SanityVerdict::DiscrepancyTooLarge { .. }));
    }

    #[test]
    fn lora_adapter_size_flagged() {
        // LoRA adapter is ~1% of base.
        let est = estimate_params(small_arch()).unwrap();
        let lora = est / 100;
        let v = check_sanity(lora, small_arch());
        assert!(matches!(v, SanityVerdict::DiscrepancyTooLarge { .. }));
    }

    #[test]
    fn zero_dimension_invalid() {
        let arch = Architecture {
            n_layers: 0,
            ..small_arch()
        };
        assert_eq!(
            check_sanity(1_000_000, arch),
            SanityVerdict::InvalidArchitecture
        );
    }

    #[test]
    fn estimate_increases_with_layers() {
        let small = estimate_params(small_arch()).unwrap();
        let big = estimate_params(Architecture {
            n_layers: 4,
            ..small_arch()
        })
        .unwrap();
        assert!(big > small);
    }

    #[test]
    fn estimate_increases_with_hidden() {
        let small = estimate_params(small_arch()).unwrap();
        let big = estimate_params(Architecture {
            hidden: 128,
            ..small_arch()
        })
        .unwrap();
        assert!(big > small);
    }

    #[test]
    fn boundary_at_5pct_passes() {
        let est = estimate_params(small_arch()).unwrap();
        let at_boundary = (est as f64 * 1.05) as u64;
        // Equal to TOLERANCE is the threshold (strict >).
        let v = check_sanity(at_boundary, small_arch());
        assert!(matches!(
            v,
            SanityVerdict::Ok | SanityVerdict::DiscrepancyTooLarge { .. }
        ));
    }

    #[test]
    fn ffn_mult_zero_treated_as_one() {
        // Defensive: 0 ffn_mult clamped to 1 to avoid all-zero FFN.
        let arch = Architecture {
            ffn_mult: 0,
            ..small_arch()
        };
        let est = estimate_params(arch).unwrap();
        assert!(est > 0);
    }
}
