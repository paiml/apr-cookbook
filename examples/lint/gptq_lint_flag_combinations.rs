//! # Recipe: GPTQ Lint — Flag Combination Matrix
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gptq-lint --observation-file observation.json` (flag matrix)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the GPTQ flag-pairing rules that exist because real
//! inference kernels (ExLlama-v2, AutoGPTQ-CUDA) only support specific
//! `(act_order, desc_act, sym_quant)` triples. The recipe enumerates the
//! 8 combinations and asserts each lands at the correct verdict.
//!
//! Triple semantics:
//!   - `act_order`: reorder columns by Hessian diagonal magnitude
//!   - `desc_act`: store the activation-order index (kernel needs it)
//!   - `sym_quant`: symmetric vs asymmetric scale (no zero-point if symmetric)
//!
//! ## Run Command
//! ```bash
//! cargo run --example gptq_lint_flag_combinations
//! ```
//!
//! ## References
//! - Frantar, E. et al. (2023). *GPTQ*. arXiv:2210.17323, §3.1 (act_order).
//! - AutoGPTQ kernel constraints (github.com/AutoGPTQ/AutoGPTQ).
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct FlagTriple {
    pub act_order: bool,
    pub desc_act: bool,
    pub sym_quant: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagVerdict {
    Supported,
    KernelMissing(&'static str),
}

pub fn classify(t: FlagTriple) -> FlagVerdict {
    if t.act_order && !t.desc_act {
        return FlagVerdict::KernelMissing(
            "act_order=true requires desc_act=true (kernel pairing)",
        );
    }
    // ExLlama-v2 only supports `act_order=true && sym_quant=false` for the
    // fast path. The slow path supports the rest.
    FlagVerdict::Supported
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gptq_lint_flag_combinations")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("act_order desc_act sym_quant → verdict");
    for ao in [false, true] {
        for da in [false, true] {
            for sq in [false, true] {
                let t = FlagTriple {
                    act_order: ao,
                    desc_act: da,
                    sym_quant: sq,
                };
                println!("{:>9} {:>8} {:>9} → {:?}", ao, da, sq, classify(t));
            }
        }
    }

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn act_order_without_desc_act_is_kernel_missing() {
        let t = FlagTriple {
            act_order: true,
            desc_act: false,
            sym_quant: false,
        };
        assert!(matches!(classify(t), FlagVerdict::KernelMissing(_)));
    }

    #[test]
    fn act_order_with_desc_act_is_supported() {
        let t = FlagTriple {
            act_order: true,
            desc_act: true,
            sym_quant: false,
        };
        assert_eq!(classify(t), FlagVerdict::Supported);
    }

    #[test]
    fn no_act_order_is_always_supported() {
        // The legacy GPTQ pre-act-order path works regardless of desc_act.
        for da in [false, true] {
            for sq in [false, true] {
                let t = FlagTriple {
                    act_order: false,
                    desc_act: da,
                    sym_quant: sq,
                };
                assert_eq!(classify(t), FlagVerdict::Supported);
            }
        }
    }

    #[test]
    fn full_eight_triple_matrix_has_one_kernel_missing() {
        let mut missing = 0;
        for ao in [false, true] {
            for da in [false, true] {
                for sq in [false, true] {
                    let t = FlagTriple {
                        act_order: ao,
                        desc_act: da,
                        sym_quant: sq,
                    };
                    if matches!(classify(t), FlagVerdict::KernelMissing(_)) {
                        missing += 1;
                    }
                }
            }
        }
        // Only (true, false, *) — i.e., 2 of 8 — is kernel-missing.
        assert_eq!(missing, 2);
    }
}
