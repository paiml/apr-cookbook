//! # SIMD FMA Fusion Eligibility Gate
//!
//! `a * b + c` can be fused into a single `fmadd` instruction (one
//! rounding step, lower latency) only when:
//! - the target CPU supports the relevant FMA ISA (FMA3/FMA4/NEON-FMA),
//! - the precision is f32/f64 (no fused half on most CPUs),
//! - strict-rounding is not required by the user (fmadd has 1 round
//!   instead of mul-then-add 2 rounds).
//!
//! This recipe builds the gate.
//!
//! Demonstrates the **SIMD.11** recipe for PMAT-134 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: IEEE 754-2008 fused multiply-add definition.
//!
//! Run with: cargo run --example simd_fma_fusion_gate
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIsa {
    Fma3,
    Fma4,
    NeonFma,
    NoFma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
    F64,
}

#[derive(Debug, PartialEq)]
pub enum FuseVerdict {
    Fuse,
    Skip { reason: SkipReason },
}

#[derive(Debug, PartialEq, Eq)]
pub enum SkipReason {
    NoIsaSupport,
    StrictRoundingRequired,
    UnsupportedHalfPrecision,
}

pub fn decide(isa: CpuIsa, precision: Precision, strict_rounding: bool) -> FuseVerdict {
    if strict_rounding {
        return FuseVerdict::Skip {
            reason: SkipReason::StrictRoundingRequired,
        };
    }
    if isa == CpuIsa::NoFma {
        return FuseVerdict::Skip {
            reason: SkipReason::NoIsaSupport,
        };
    }
    if precision == Precision::F16 {
        return FuseVerdict::Skip {
            reason: SkipReason::UnsupportedHalfPrecision,
        };
    }
    FuseVerdict::Fuse
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_fma_fusion_gate")?;

    let cases = [
        (CpuIsa::Fma3, Precision::F32, false),
        (CpuIsa::Fma3, Precision::F32, true),
        (CpuIsa::NoFma, Precision::F32, false),
        (CpuIsa::Fma3, Precision::F16, false),
        (CpuIsa::NeonFma, Precision::F64, false),
    ];
    for (isa, prec, strict) in cases {
        println!(
            "{isa:?} / {prec:?} / strict={strict} → {:?}",
            decide(isa, prec, strict)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fma3_f32_default_fuses() {
        assert_eq!(
            decide(CpuIsa::Fma3, Precision::F32, false),
            FuseVerdict::Fuse
        );
    }

    #[test]
    fn fma3_f64_fuses() {
        assert_eq!(
            decide(CpuIsa::Fma3, Precision::F64, false),
            FuseVerdict::Fuse
        );
    }

    #[test]
    fn neon_fuses() {
        assert_eq!(
            decide(CpuIsa::NeonFma, Precision::F32, false),
            FuseVerdict::Fuse
        );
    }

    #[test]
    fn no_isa_skipped() {
        let v = decide(CpuIsa::NoFma, Precision::F32, false);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::NoIsaSupport
            }
        );
    }

    #[test]
    fn strict_rounding_skipped() {
        let v = decide(CpuIsa::Fma3, Precision::F32, true);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::StrictRoundingRequired
            }
        );
    }

    #[test]
    fn half_precision_skipped() {
        let v = decide(CpuIsa::Fma3, Precision::F16, false);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::UnsupportedHalfPrecision
            }
        );
    }

    #[test]
    fn strict_rounding_takes_precedence_over_isa() {
        // Strict-rounding requirement is checked first → skipped even on
        // capable CPU.
        let v = decide(CpuIsa::Fma4, Precision::F32, true);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::StrictRoundingRequired
            }
        );
    }

    #[test]
    fn fma4_f64_fuses() {
        assert_eq!(
            decide(CpuIsa::Fma4, Precision::F64, false),
            FuseVerdict::Fuse
        );
    }

    #[test]
    fn no_fma_with_strict_strict_wins() {
        // Strict reason reported even when there's also no ISA.
        let v = decide(CpuIsa::NoFma, Precision::F32, true);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::StrictRoundingRequired
            }
        );
    }

    #[test]
    fn neon_f16_skipped() {
        let v = decide(CpuIsa::NeonFma, Precision::F16, false);
        assert_eq!(
            v,
            FuseVerdict::Skip {
                reason: SkipReason::UnsupportedHalfPrecision
            }
        );
    }
}
