//! # WASM Imported-Function Count Gate
//!
//! Each WASM-imported function call crosses the JS/host boundary,
//! ~200 ns minimum on V8 (vs ~5 ns for in-WASM call). Lots of imports
//! = lots of bridge crossings = slow.
//!
//! Tier rule:
//!   ≤ 50 imports:   Lean
//!   51-200:         Acceptable
//!   201-500:        Heavy
//!   > 500:          Excessive (refactor: batch calls, use shared memory)
//!
//! Demonstrates the **WASM.19** recipe for PMAT-142 (wasm round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: V8 wasm interop overhead measurements (WebAssembly.org docs).
//!
//! Run with: cargo run --example wasm_imported_function_count
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportTier {
    Lean,
    Acceptable,
    Heavy,
    Excessive,
}

#[derive(Debug, PartialEq)]
pub enum ImportVerdict {
    Ok {
        tier: ImportTier,
        estimated_bridge_overhead_us: u64,
    },
    EmptyImports,
}

const BRIDGE_OVERHEAD_NS: u64 = 200;

pub fn classify(import_count: u32, expected_calls_per_request: u32) -> ImportVerdict {
    if import_count == 0 {
        return ImportVerdict::EmptyImports;
    }
    let tier = match import_count {
        0..=50 => ImportTier::Lean,
        51..=200 => ImportTier::Acceptable,
        201..=500 => ImportTier::Heavy,
        _ => ImportTier::Excessive,
    };
    let overhead_ns = u64::from(expected_calls_per_request) * BRIDGE_OVERHEAD_NS;
    ImportVerdict::Ok {
        tier,
        estimated_bridge_overhead_us: overhead_ns / 1_000,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_imported_function_count")?;

    println!("10 imports, 5 calls: {:?}", classify(10, 5));
    println!("100 imports, 50 calls: {:?}", classify(100, 50));
    println!("300 imports, 200 calls: {:?}", classify(300, 200));
    println!("700 imports, 1000 calls: {:?}", classify(700, 1000));
    println!("zero imports: {:?}", classify(0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_count_lean() {
        let v = classify(10, 5);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Lean);
        }
    }

    #[test]
    fn medium_acceptable() {
        let v = classify(100, 50);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Acceptable);
        }
    }

    #[test]
    fn high_heavy() {
        let v = classify(300, 200);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Heavy);
        }
    }

    #[test]
    fn excessive_above_500() {
        let v = classify(700, 1000);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Excessive);
        }
    }

    #[test]
    fn zero_rejected() {
        assert_eq!(classify(0, 0), ImportVerdict::EmptyImports);
    }

    #[test]
    fn boundary_at_50_lean() {
        let v = classify(50, 1);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Lean);
        }
    }

    #[test]
    fn boundary_at_51_acceptable() {
        let v = classify(51, 1);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Acceptable);
        }
    }

    #[test]
    fn boundary_at_500_heavy() {
        let v = classify(500, 1);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Heavy);
        }
    }

    #[test]
    fn boundary_at_501_excessive() {
        let v = classify(501, 1);
        if let ImportVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ImportTier::Excessive);
        }
    }

    #[test]
    fn overhead_proportional_to_calls() {
        let v_low = classify(10, 100);
        let v_high = classify(10, 1000);
        if let (
            ImportVerdict::Ok {
                estimated_bridge_overhead_us: low,
                ..
            },
            ImportVerdict::Ok {
                estimated_bridge_overhead_us: high,
                ..
            },
        ) = (v_low, v_high)
        {
            assert_eq!(high / low, 10);
        }
    }

    #[test]
    fn overhead_independent_of_import_count() {
        // Bridge overhead per call is constant regardless of import count.
        let v_few = classify(10, 100);
        let v_many = classify(500, 100);
        if let (
            ImportVerdict::Ok {
                estimated_bridge_overhead_us: a,
                ..
            },
            ImportVerdict::Ok {
                estimated_bridge_overhead_us: b,
                ..
            },
        ) = (v_few, v_many)
        {
            assert_eq!(a, b);
        }
    }
}
