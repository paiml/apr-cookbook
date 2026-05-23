//! # WASM SIMD128 Runtime Dispatch
//!
//! WebAssembly SIMD128 is a feature-test: not all browsers/runtimes
//! enable it (Safari < 16.4, older mobile WebViews). Strategy:
//! detect at module instantiation, compile a SIMD-using path or a
//! scalar-fallback path, expose them under one symbol. This recipe
//! builds the dispatch picker.
//!
//! Demonstrates the **WASM.12** recipe for PMAT-134 (wasm coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly fixed-width SIMD proposal (v8/v128).
//!
//! Run with: cargo run --example wasm_simd128_dispatch
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPath {
    Simd128,
    Scalar,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok {
        path: DispatchPath,
        expected_speedup: f64,
    },
    BothPathsMissing,
}

pub struct RuntimeProbe {
    pub simd128_available: bool,
    pub workload_vectorizable: bool,
    pub min_speedup_threshold: f64,
}

pub fn pick(probe: &RuntimeProbe, scalar_available: bool) -> DispatchVerdict {
    if !scalar_available && !probe.simd128_available {
        return DispatchVerdict::BothPathsMissing;
    }
    let want_simd = probe.simd128_available
        && probe.workload_vectorizable
        && probe.min_speedup_threshold >= 1.0;
    let path = if want_simd && probe.simd128_available {
        DispatchPath::Simd128
    } else {
        DispatchPath::Scalar
    };
    let expected_speedup = if path == DispatchPath::Simd128 {
        4.0_f64.max(probe.min_speedup_threshold)
    } else {
        1.0
    };
    DispatchVerdict::Ok {
        path,
        expected_speedup,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_simd128_dispatch")?;

    let probe = RuntimeProbe {
        simd128_available: true,
        workload_vectorizable: true,
        min_speedup_threshold: 2.0,
    };
    println!("simd path: {:?}", pick(&probe, true));

    let no_simd = RuntimeProbe {
        simd128_available: false,
        ..probe
    };
    println!("scalar fallback: {:?}", pick(&no_simd, true));

    let both_gone = RuntimeProbe {
        simd128_available: false,
        ..probe
    };
    println!("both missing: {:?}", pick(&both_gone, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(simd: bool, vec: bool) -> RuntimeProbe {
        RuntimeProbe {
            simd128_available: simd,
            workload_vectorizable: vec,
            min_speedup_threshold: 2.0,
        }
    }

    #[test]
    fn dispatch_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_capability_picks_simd() {
        let v = pick(&probe(true, true), true);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                path: DispatchPath::Simd128,
                ..
            }
        ));
    }

    #[test]
    fn missing_simd_falls_back_to_scalar() {
        let v = pick(&probe(false, true), true);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                path: DispatchPath::Scalar,
                ..
            }
        ));
    }

    #[test]
    fn non_vectorizable_picks_scalar() {
        let v = pick(&probe(true, false), true);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                path: DispatchPath::Scalar,
                ..
            }
        ));
    }

    #[test]
    fn both_missing_rejected() {
        let v = pick(&probe(false, false), false);
        assert_eq!(v, DispatchVerdict::BothPathsMissing);
    }

    #[test]
    fn scalar_path_speedup_one() {
        if let DispatchVerdict::Ok {
            expected_speedup, ..
        } = pick(&probe(false, true), true)
        {
            assert_eq!(expected_speedup, 1.0);
        }
    }

    #[test]
    fn simd_path_speedup_at_least_four() {
        if let DispatchVerdict::Ok {
            expected_speedup, ..
        } = pick(&probe(true, true), true)
        {
            assert!(expected_speedup >= 4.0);
        }
    }

    #[test]
    fn high_threshold_overrides_default_speedup() {
        let p = RuntimeProbe {
            simd128_available: true,
            workload_vectorizable: true,
            min_speedup_threshold: 6.0,
        };
        if let DispatchVerdict::Ok {
            expected_speedup, ..
        } = pick(&p, true)
        {
            assert_eq!(expected_speedup, 6.0);
        }
    }

    #[test]
    fn threshold_below_one_skips_simd() {
        let p = RuntimeProbe {
            simd128_available: true,
            workload_vectorizable: true,
            min_speedup_threshold: 0.5,
        };
        let v = pick(&p, true);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                path: DispatchPath::Scalar,
                ..
            }
        ));
    }

    #[test]
    fn simd_only_no_scalar_still_picks_simd() {
        // No scalar fallback exists, but SIMD is available.
        let v = pick(&probe(true, true), false);
        assert!(matches!(
            v,
            DispatchVerdict::Ok {
                path: DispatchPath::Simd128,
                ..
            }
        ));
    }
}
