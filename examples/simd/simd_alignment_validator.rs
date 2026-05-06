//! # SIMD Pointer Alignment Validator
//!
//! SIMD intrinsics require aligned data: SSE = 16 bytes, AVX/AVX2 = 32
//! bytes, AVX-512 = 64 bytes, NEON = 16 bytes. Mis-aligned access on
//! some ISAs faults; on others it's a slow scalar fallback. This recipe
//! validates pointer alignment + recommends a load instruction.
//!
//! Demonstrates the **SIMD.5** recipe for PMAT-123 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel Intrinsics Guide; ARM NEON Programmer's Guide.
//!
//! Run with: cargo run --example simd_alignment_validator
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdIsa {
    Sse,
    Avx2,
    Avx512,
    Neon,
}

impl SimdIsa {
    pub fn required_bytes(self) -> usize {
        match self {
            SimdIsa::Sse | SimdIsa::Neon => 16,
            SimdIsa::Avx2 => 32,
            SimdIsa::Avx512 => 64,
        }
    }

    pub fn aligned_load_intrinsic(self) -> &'static str {
        match self {
            SimdIsa::Sse => "_mm_load_ps",
            SimdIsa::Avx2 => "_mm256_load_ps",
            SimdIsa::Avx512 => "_mm512_load_ps",
            SimdIsa::Neon => "vld1q_f32",
        }
    }

    pub fn unaligned_load_intrinsic(self) -> &'static str {
        match self {
            SimdIsa::Sse => "_mm_loadu_ps",
            SimdIsa::Avx2 => "_mm256_loadu_ps",
            SimdIsa::Avx512 => "_mm512_loadu_ps",
            SimdIsa::Neon => "vld1q_f32_unaligned",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum AlignmentVerdict {
    AlignedUseFastLoad {
        intrinsic: &'static str,
    },
    UnalignedUseSlowLoad {
        intrinsic: &'static str,
        misalignment_bytes: usize,
    },
    NullPointer,
}

pub fn classify(ptr_addr: usize, isa: SimdIsa) -> AlignmentVerdict {
    if ptr_addr == 0 {
        return AlignmentVerdict::NullPointer;
    }
    let req = isa.required_bytes();
    let mis = ptr_addr % req;
    if mis == 0 {
        AlignmentVerdict::AlignedUseFastLoad {
            intrinsic: isa.aligned_load_intrinsic(),
        }
    } else {
        AlignmentVerdict::UnalignedUseSlowLoad {
            intrinsic: isa.unaligned_load_intrinsic(),
            misalignment_bytes: mis,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_alignment_validator")?;

    for isa in [SimdIsa::Sse, SimdIsa::Avx2, SimdIsa::Avx512, SimdIsa::Neon] {
        for addr in [0x1000usize, 0x1010, 0x1020, 0x1040] {
            println!("ISA={isa:?} addr={addr:#x}  →  {:?}", classify(addr, isa));
        }
    }
    println!("null: {:?}", classify(0, SimdIsa::Avx2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn aligned_address_uses_fast_load() {
        // 0x1000 is aligned to 32 bytes for AVX2.
        let v = classify(0x1000, SimdIsa::Avx2);
        assert!(matches!(v, AlignmentVerdict::AlignedUseFastLoad { .. }));
    }

    #[test]
    fn misaligned_address_uses_slow_load() {
        // 0x1010 is 16-byte aligned but not 32-byte → slow load for AVX2.
        let v = classify(0x1010, SimdIsa::Avx2);
        assert!(matches!(
            v,
            AlignmentVerdict::UnalignedUseSlowLoad {
                misalignment_bytes: 16,
                ..
            }
        ));
    }

    #[test]
    fn null_pointer_rejected() {
        assert_eq!(classify(0, SimdIsa::Avx2), AlignmentVerdict::NullPointer);
    }

    #[test]
    fn sse_requires_16_bytes() {
        assert_eq!(SimdIsa::Sse.required_bytes(), 16);
        assert!(matches!(
            classify(0x1010, SimdIsa::Sse),
            AlignmentVerdict::AlignedUseFastLoad { .. }
        ));
    }

    #[test]
    fn avx512_requires_64_bytes() {
        assert_eq!(SimdIsa::Avx512.required_bytes(), 64);
        // 0x1020 is 32-aligned but not 64-aligned → slow.
        let v = classify(0x1020, SimdIsa::Avx512);
        assert!(matches!(v, AlignmentVerdict::UnalignedUseSlowLoad { .. }));
        // 0x1040 IS 64-aligned.
        let v2 = classify(0x1040, SimdIsa::Avx512);
        assert!(matches!(v2, AlignmentVerdict::AlignedUseFastLoad { .. }));
    }

    #[test]
    fn neon_requires_16_bytes() {
        assert_eq!(SimdIsa::Neon.required_bytes(), 16);
        assert!(matches!(
            classify(0x100, SimdIsa::Neon),
            AlignmentVerdict::AlignedUseFastLoad { .. }
        ));
    }

    #[test]
    fn misalignment_bytes_reported_correctly() {
        // 0x1004 is 4 bytes off 16-byte alignment.
        let v = classify(0x1004, SimdIsa::Sse);
        assert!(matches!(
            v,
            AlignmentVerdict::UnalignedUseSlowLoad {
                misalignment_bytes: 4,
                ..
            }
        ));
    }

    #[test]
    fn intrinsic_names_match_isa() {
        if let AlignmentVerdict::AlignedUseFastLoad { intrinsic } = classify(0x1000, SimdIsa::Avx2)
        {
            assert_eq!(intrinsic, "_mm256_load_ps");
        }
    }
}
