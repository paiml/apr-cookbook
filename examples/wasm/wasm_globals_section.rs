//! # WASM Globals-Section Size Estimator
//!
//! Globals section: each entry has type byte + mut byte + init expr
//! (~5-9 bytes for i32 const, more for f64). Picker estimates total
//! size + tier (Slim/Average/Bloated).
//!
//! Demonstrates the **WASM.21** recipe for PMAT-146 (wasm round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core Spec § 5.5.10 (Globals Section).
//!
//! Run with: cargo run --example wasm_globals_section
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalType {
    I32,
    I64,
    F32,
    F64,
    Externref,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalsTier {
    Slim,
    Average,
    Bloated,
}

#[derive(Debug, PartialEq)]
pub enum GlobalsVerdict {
    Ok { total_bytes: u64, tier: GlobalsTier },
    EmptyGlobals,
}

const PREAMBLE_BYTES: u64 = 4;
const SLIM_LIMIT: u64 = 512;
const AVG_LIMIT: u64 = 8 * 1024;

pub fn estimate(globals: &[GlobalType]) -> GlobalsVerdict {
    if globals.is_empty() {
        return GlobalsVerdict::EmptyGlobals;
    }
    let mut total = PREAMBLE_BYTES;
    for g in globals {
        let bytes = match g {
            GlobalType::I32 | GlobalType::F32 => 7,
            GlobalType::I64 | GlobalType::F64 => 11,
            GlobalType::Externref => 6,
        };
        total += bytes;
    }
    let tier = if total <= SLIM_LIMIT {
        GlobalsTier::Slim
    } else if total <= AVG_LIMIT {
        GlobalsTier::Average
    } else {
        GlobalsTier::Bloated
    };
    GlobalsVerdict::Ok {
        total_bytes: total,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_globals_section")?;

    println!(
        "5 mixed: {:?}",
        estimate(&[
            GlobalType::I32,
            GlobalType::I64,
            GlobalType::F64,
            GlobalType::F32,
            GlobalType::Externref
        ])
    );

    let many: Vec<GlobalType> = (0..1000).map(|_| GlobalType::I32).collect();
    println!("1000 i32: {:?}", estimate(&many));

    let huge: Vec<GlobalType> = (0..2000).map(|_| GlobalType::F64).collect();
    println!("2000 f64: {:?}", estimate(&huge));

    println!("empty: {:?}", estimate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(estimate(&[]), GlobalsVerdict::EmptyGlobals);
    }

    #[test]
    fn small_set_slim() {
        let v = estimate(&[GlobalType::I32; 10]);
        if let GlobalsVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, GlobalsTier::Slim);
        }
    }

    #[test]
    fn medium_set_average() {
        // ~1000 i32 = 7000 bytes → Average.
        let many: Vec<GlobalType> = (0..1000).map(|_| GlobalType::I32).collect();
        if let GlobalsVerdict::Ok { tier, .. } = estimate(&many) {
            assert_eq!(tier, GlobalsTier::Average);
        }
    }

    #[test]
    fn large_set_bloated() {
        // 2000 f64 = 22 000 bytes → Bloated.
        let many: Vec<GlobalType> = (0..2000).map(|_| GlobalType::F64).collect();
        if let GlobalsVerdict::Ok { tier, .. } = estimate(&many) {
            assert_eq!(tier, GlobalsTier::Bloated);
        }
    }

    #[test]
    fn i64_costs_more_than_i32() {
        let i32_v = estimate(&[GlobalType::I32]);
        let i64_v = estimate(&[GlobalType::I64]);
        if let (
            GlobalsVerdict::Ok { total_bytes: a, .. },
            GlobalsVerdict::Ok { total_bytes: b, .. },
        ) = (i32_v, i64_v)
        {
            assert!(b > a);
        }
    }

    #[test]
    fn f32_same_size_as_i32() {
        let i32_v = estimate(&[GlobalType::I32]);
        let f32_v = estimate(&[GlobalType::F32]);
        assert_eq!(i32_v, f32_v);
    }

    #[test]
    fn includes_preamble() {
        // Even one i32 entry has preamble.
        if let GlobalsVerdict::Ok { total_bytes, .. } = estimate(&[GlobalType::I32]) {
            assert_eq!(total_bytes, PREAMBLE_BYTES + 7);
        }
    }

    #[test]
    fn externref_costs_6() {
        if let GlobalsVerdict::Ok { total_bytes, .. } = estimate(&[GlobalType::Externref]) {
            assert_eq!(total_bytes, PREAMBLE_BYTES + 6);
        }
    }

    #[test]
    fn boundary_at_512_bytes_slim() {
        // 72 i32 globals = 4 + 72×7 = 508 bytes → still Slim.
        let many: Vec<GlobalType> = (0..72).map(|_| GlobalType::I32).collect();
        if let GlobalsVerdict::Ok { tier, .. } = estimate(&many) {
            assert_eq!(tier, GlobalsTier::Slim);
        }
    }

    #[test]
    fn just_above_slim_limit_average() {
        // 73 i32 = 4 + 511 = 515 bytes → Average.
        let many: Vec<GlobalType> = (0..73).map(|_| GlobalType::I32).collect();
        if let GlobalsVerdict::Ok { tier, .. } = estimate(&many) {
            assert_eq!(tier, GlobalsTier::Average);
        }
    }
}
