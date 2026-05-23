//! # SIMD Alignment Strategy Picker
//!
//! Three approaches for aligned SIMD load/store on a base pointer:
//!   AlignedOnly: assert pointer is aligned; faster but UB if not
//!   Unaligned: use vmovups/loadu; ~equal cost on modern CPUs
//!   PrologueEpilogue: scalar prologue to reach alignment, then aligned
//!     SIMD body, scalar epilogue for tail
//!
//! Picker rules:
//!   - Already aligned + length ≥ 64: AlignedOnly
//!   - Misaligned + length ≥ 1024: PrologueEpilogue (worth the prologue cost)
//!   - Otherwise: Unaligned
//!
//! Demonstrates the **SIMD.13** recipe for PMAT-138 (simd round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel optimization manual § alignment for SSE/AVX.
//!
//! Run with: cargo run --example simd_alignment_strategy
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const VECTOR_BYTES: u64 = 64;
const PROLOGUE_THRESHOLD_BYTES: u64 = 1024;
const ALIGNED_BODY_MIN: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    AlignedOnly,
    Unaligned,
    PrologueEpilogue {
        prologue_bytes: u64,
        epilogue_bytes: u64,
    },
}

#[derive(Debug, PartialEq)]
pub enum StrategyVerdict {
    Ok(Strategy),
    InvalidLength,
}

pub fn pick(base_offset_mod_vector: u64, length_bytes: u64) -> StrategyVerdict {
    if length_bytes == 0 {
        return StrategyVerdict::InvalidLength;
    }
    let aligned = base_offset_mod_vector == 0;
    if aligned && length_bytes >= ALIGNED_BODY_MIN {
        return StrategyVerdict::Ok(Strategy::AlignedOnly);
    }
    if !aligned && length_bytes >= PROLOGUE_THRESHOLD_BYTES {
        let prologue = VECTOR_BYTES - base_offset_mod_vector;
        let body = length_bytes - prologue;
        let epilogue = body % VECTOR_BYTES;
        return StrategyVerdict::Ok(Strategy::PrologueEpilogue {
            prologue_bytes: prologue,
            epilogue_bytes: epilogue,
        });
    }
    StrategyVerdict::Ok(Strategy::Unaligned)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_alignment_strategy")?;

    println!("aligned, 1024 bytes: {:?}", pick(0, 1024));
    println!("misaligned by 8, 100 bytes: {:?}", pick(8, 100));
    println!("misaligned by 8, 4096 bytes: {:?}", pick(8, 4096));
    println!("aligned, 32 bytes (too short): {:?}", pick(0, 32));
    println!("invalid length: {:?}", pick(0, 0));
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
    fn aligned_long_body_aligned_only() {
        let v = pick(0, 1024);
        assert_eq!(v, StrategyVerdict::Ok(Strategy::AlignedOnly));
    }

    #[test]
    fn misaligned_short_body_unaligned() {
        let v = pick(8, 100);
        assert_eq!(v, StrategyVerdict::Ok(Strategy::Unaligned));
    }

    #[test]
    fn misaligned_long_body_prologue() {
        let v = pick(8, 4096);
        assert!(matches!(
            v,
            StrategyVerdict::Ok(Strategy::PrologueEpilogue { .. })
        ));
    }

    #[test]
    fn aligned_short_body_unaligned() {
        // < ALIGNED_BODY_MIN → Unaligned.
        let v = pick(0, 32);
        assert_eq!(v, StrategyVerdict::Ok(Strategy::Unaligned));
    }

    #[test]
    fn invalid_length_zero_rejected() {
        assert_eq!(pick(0, 0), StrategyVerdict::InvalidLength);
    }

    #[test]
    fn prologue_brings_to_alignment() {
        // base_offset_mod_vector = 8, vector = 64 → prologue = 56.
        if let StrategyVerdict::Ok(Strategy::PrologueEpilogue { prologue_bytes, .. }) =
            pick(8, 4096)
        {
            assert_eq!(prologue_bytes, 56);
        }
    }

    #[test]
    fn epilogue_handles_remainder() {
        // After 56 byte prologue: 4096 - 56 = 4040 body.
        // 4040 % 64 = 8 epilogue.
        if let StrategyVerdict::Ok(Strategy::PrologueEpilogue { epilogue_bytes, .. }) =
            pick(8, 4096)
        {
            assert_eq!(epilogue_bytes, 8);
        }
    }

    #[test]
    fn aligned_at_min_body_aligned_only() {
        let v = pick(0, 64);
        assert_eq!(v, StrategyVerdict::Ok(Strategy::AlignedOnly));
    }

    #[test]
    fn just_below_prologue_threshold_unaligned() {
        let v = pick(8, 1023);
        assert_eq!(v, StrategyVerdict::Ok(Strategy::Unaligned));
    }

    #[test]
    fn at_prologue_threshold_starts_prologue() {
        let v = pick(8, 1024);
        assert!(matches!(
            v,
            StrategyVerdict::Ok(Strategy::PrologueEpilogue { .. })
        ));
    }
}
