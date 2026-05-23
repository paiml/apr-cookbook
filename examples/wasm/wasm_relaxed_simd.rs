//! # WASM Relaxed-SIMD Dispatcher
//!
//! Relaxed SIMD allows implementation-defined behavior for some ops
//! (e.g., min/max NaN handling) → faster on most hardware but less
//! deterministic. Picker: enable for non-IEEE-strict workloads,
//! disable for cross-platform reproducibility.
//!
//! Demonstrates the **WASM.23** recipe for PMAT-151 (wasm round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Relaxed SIMD proposal.
//!
//! Run with: cargo run --example wasm_relaxed_simd
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdMode {
    StrictDeterministic,
    RelaxedFast,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok {
        mode: SimdMode,
        expected_speedup: f64,
    },
    UnsupportedRuntime,
}

pub fn pick(
    runtime_supports_relaxed: bool,
    requires_bit_exact_reproducibility: bool,
) -> DispatchVerdict {
    if !runtime_supports_relaxed && !requires_bit_exact_reproducibility {
        return DispatchVerdict::UnsupportedRuntime;
    }
    let mode = if requires_bit_exact_reproducibility || !runtime_supports_relaxed {
        SimdMode::StrictDeterministic
    } else {
        SimdMode::RelaxedFast
    };
    let expected_speedup = match mode {
        SimdMode::StrictDeterministic => 1.0,
        SimdMode::RelaxedFast => 1.2,
    };
    DispatchVerdict::Ok {
        mode,
        expected_speedup,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_relaxed_simd")?;

    println!("relaxed runtime, no strict: {:?}", pick(true, false));
    println!("relaxed runtime, strict needed: {:?}", pick(true, true));
    println!(
        "strict only runtime, strict needed: {:?}",
        pick(false, true)
    );
    println!("strict only, no strict needed: {:?}", pick(false, false));
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
    fn relaxed_when_supported_no_strict() {
        let v = pick(true, false);
        if let DispatchVerdict::Ok { mode, .. } = v {
            assert_eq!(mode, SimdMode::RelaxedFast);
        }
    }

    #[test]
    fn strict_when_required() {
        let v = pick(true, true);
        if let DispatchVerdict::Ok { mode, .. } = v {
            assert_eq!(mode, SimdMode::StrictDeterministic);
        }
    }

    #[test]
    fn strict_only_runtime_strict_mode() {
        let v = pick(false, true);
        if let DispatchVerdict::Ok { mode, .. } = v {
            assert_eq!(mode, SimdMode::StrictDeterministic);
        }
    }

    #[test]
    fn unsupported_runtime_no_strict_rejected() {
        assert_eq!(pick(false, false), DispatchVerdict::UnsupportedRuntime);
    }

    #[test]
    fn relaxed_speedup_higher() {
        let relaxed = pick(true, false);
        let strict = pick(true, true);
        if let (
            DispatchVerdict::Ok {
                expected_speedup: r,
                ..
            },
            DispatchVerdict::Ok {
                expected_speedup: s,
                ..
            },
        ) = (relaxed, strict)
        {
            assert!(r > s);
        }
    }

    #[test]
    fn strict_speedup_one() {
        let v = pick(true, true);
        if let DispatchVerdict::Ok {
            expected_speedup, ..
        } = v
        {
            assert_eq!(expected_speedup, 1.0);
        }
    }

    #[test]
    fn relaxed_speedup_at_least_one() {
        let v = pick(true, false);
        if let DispatchVerdict::Ok {
            expected_speedup, ..
        } = v
        {
            assert!(expected_speedup >= 1.0);
        }
    }

    #[test]
    fn strict_required_overrides_runtime() {
        // Even with relaxed runtime, strict requirement → strict mode.
        let v = pick(true, true);
        if let DispatchVerdict::Ok { mode, .. } = v {
            assert_eq!(mode, SimdMode::StrictDeterministic);
        }
    }

    #[test]
    fn workflow_picks_consistent_mode() {
        // Calling twice with same input → same result.
        let a = pick(true, false);
        let b = pick(true, false);
        assert_eq!(a, b);
    }

    #[test]
    fn ok_returned_for_each_valid_combo() {
        let combos = [(true, true), (true, false), (false, true)];
        for (rt, strict) in combos {
            assert!(matches!(pick(rt, strict), DispatchVerdict::Ok { .. }));
        }
    }
}
