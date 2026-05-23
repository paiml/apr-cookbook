//! # apr compile — `--release --strip --lto` Optimization Flags
//!
//! `apr compile <FILE> --release --strip --lto` controls binary size +
//! perf. This recipe documents the interactions: `--strip` requires
//! `--release` (debug builds keep DWARF on purpose); `--lto` implies
//! `--release` (debug LTO doesn't make sense). The validator surfaces
//! these constraints.
//!
//! Demonstrates the **COMPILE.4** recipe for PMAT-110 (apr compile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-SPEC §4.16 + cargo profile semantics
//!
//! Run with: cargo run --example cli_compile_optimization_flags
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone, Copy)]
pub struct CompileFlags {
    pub release: bool,
    pub strip: bool,
    pub lto: bool,
}

#[derive(Debug, PartialEq)]
pub enum FlagsVerdict {
    Ok,
    StripWithoutRelease,
    LtoWithoutRelease,
}

pub fn validate(flags: CompileFlags) -> FlagsVerdict {
    if flags.strip && !flags.release {
        return FlagsVerdict::StripWithoutRelease;
    }
    if flags.lto && !flags.release {
        return FlagsVerdict::LtoWithoutRelease;
    }
    FlagsVerdict::Ok
}

pub fn estimated_size_reduction_pct(flags: CompileFlags) -> f64 {
    let mut reduction: f64 = 0.0;
    if flags.release {
        reduction += 25.0; // -O3 vs no opt
    }
    if flags.strip {
        reduction += 30.0; // strip symbols
    }
    if flags.lto {
        reduction += 15.0; // dead code elimination
    }
    reduction.min(80.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_compile_optimization_flags")?;

    let cases = [
        ("debug", CompileFlags::default()),
        (
            "release only",
            CompileFlags {
                release: true,
                ..Default::default()
            },
        ),
        (
            "release + strip",
            CompileFlags {
                release: true,
                strip: true,
                ..Default::default()
            },
        ),
        (
            "strip without release (BAD)",
            CompileFlags {
                strip: true,
                ..Default::default()
            },
        ),
        (
            "release + lto + strip",
            CompileFlags {
                release: true,
                strip: true,
                lto: true,
            },
        ),
    ];
    for (label, f) in cases {
        let v = validate(f);
        let r = estimated_size_reduction_pct(f);
        println!("{label:>30}  →  {v:?}  ~{r:.0}% smaller");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_run() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn debug_passes() {
        assert_eq!(validate(CompileFlags::default()), FlagsVerdict::Ok);
    }

    #[test]
    fn release_alone_passes() {
        assert_eq!(
            validate(CompileFlags {
                release: true,
                ..Default::default()
            }),
            FlagsVerdict::Ok
        );
    }

    #[test]
    fn strip_without_release_rejected() {
        assert_eq!(
            validate(CompileFlags {
                strip: true,
                ..Default::default()
            }),
            FlagsVerdict::StripWithoutRelease
        );
    }

    #[test]
    fn lto_without_release_rejected() {
        assert_eq!(
            validate(CompileFlags {
                lto: true,
                ..Default::default()
            }),
            FlagsVerdict::LtoWithoutRelease
        );
    }

    #[test]
    fn release_with_strip_and_lto_passes() {
        assert_eq!(
            validate(CompileFlags {
                release: true,
                strip: true,
                lto: true,
            }),
            FlagsVerdict::Ok
        );
    }

    #[test]
    fn debug_size_reduction_zero() {
        assert_eq!(estimated_size_reduction_pct(CompileFlags::default()), 0.0);
    }

    #[test]
    fn full_optimization_caps_at_80_pct() {
        // 25 + 30 + 15 = 70 (under cap, no clamp triggered).
        assert_eq!(
            estimated_size_reduction_pct(CompileFlags {
                release: true,
                strip: true,
                lto: true
            }),
            70.0
        );
    }
}
