//! # SIMD Loop Unroll Factor Picker
//!
//! Loop unrolling boosts SIMD throughput by hiding pipeline latency,
//! but too aggressive a factor exhausts registers and spills to memory.
//! Rule of thumb: 4-way unroll for AVX2 (16 ymm registers), 8-way for
//! AVX-512 (32 zmm registers). This recipe picks the factor + checks
//! against register pressure.
//!
//! Demonstrates the **SIMD.6** recipe for PMAT-123 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hennessy & Patterson, Computer Architecture (6th ed.) §3.4.
//!
//! Run with: cargo run --example simd_unroll_factor_picker
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdIsa {
    Sse,
    Avx2,
    Avx512,
}

impl SimdIsa {
    pub fn architectural_registers(self) -> u32 {
        match self {
            SimdIsa::Sse => 16,
            SimdIsa::Avx2 => 16,
            SimdIsa::Avx512 => 32,
        }
    }

    pub fn typical_unroll(self) -> u32 {
        match self {
            SimdIsa::Sse => 4,
            SimdIsa::Avx2 => 4,
            SimdIsa::Avx512 => 8,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum UnrollVerdict {
    Ok { factor: u32 },
    SpillsRegisters { needed: u32, available: u32 },
    InvalidWorkPerIteration,
}

pub fn pick(isa: SimdIsa, registers_per_iteration: u32) -> UnrollVerdict {
    if registers_per_iteration == 0 {
        return UnrollVerdict::InvalidWorkPerIteration;
    }
    let architectural = isa.architectural_registers();
    let typical = isa.typical_unroll();
    let needed = registers_per_iteration.saturating_mul(typical);
    if needed > architectural {
        // Reduce factor until it fits.
        let max_factor = architectural / registers_per_iteration;
        if max_factor == 0 {
            return UnrollVerdict::SpillsRegisters {
                needed: registers_per_iteration,
                available: architectural,
            };
        }
        return UnrollVerdict::Ok { factor: max_factor };
    }
    UnrollVerdict::Ok { factor: typical }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_unroll_factor_picker")?;

    for isa in [SimdIsa::Sse, SimdIsa::Avx2, SimdIsa::Avx512] {
        for r in [1u32, 2, 4, 8, 20] {
            println!("ISA={isa:?} regs/iter={r}  →  {:?}", pick(isa, r));
        }
    }
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
    fn typical_avx2_picks_4x() {
        // 2 regs/iter × 4 unroll = 8 regs ≤ 16 architectural.
        assert_eq!(pick(SimdIsa::Avx2, 2), UnrollVerdict::Ok { factor: 4 });
    }

    #[test]
    fn typical_avx512_picks_8x() {
        assert_eq!(pick(SimdIsa::Avx512, 2), UnrollVerdict::Ok { factor: 8 });
    }

    #[test]
    fn high_pressure_reduces_factor() {
        // 5 regs/iter × 4 = 20 > 16 → reduce to 16 / 5 = 3.
        assert_eq!(pick(SimdIsa::Avx2, 5), UnrollVerdict::Ok { factor: 3 });
    }

    #[test]
    fn extreme_pressure_spills() {
        // 20 regs/iter > 16 architectural → can't fit even one iteration.
        let v = pick(SimdIsa::Avx2, 20);
        assert!(matches!(v, UnrollVerdict::SpillsRegisters { .. }));
    }

    #[test]
    fn zero_regs_invalid() {
        assert_eq!(
            pick(SimdIsa::Avx2, 0),
            UnrollVerdict::InvalidWorkPerIteration
        );
    }

    #[test]
    fn architectural_register_counts_correct() {
        assert_eq!(SimdIsa::Sse.architectural_registers(), 16);
        assert_eq!(SimdIsa::Avx2.architectural_registers(), 16);
        assert_eq!(SimdIsa::Avx512.architectural_registers(), 32);
    }

    #[test]
    fn typical_unroll_factors_correct() {
        assert_eq!(SimdIsa::Sse.typical_unroll(), 4);
        assert_eq!(SimdIsa::Avx2.typical_unroll(), 4);
        assert_eq!(SimdIsa::Avx512.typical_unroll(), 8);
    }

    #[test]
    fn avx512_with_4_regs_per_iter_picks_8x() {
        // 4 × 8 = 32 ≤ 32 architectural → fits exactly.
        assert_eq!(pick(SimdIsa::Avx512, 4), UnrollVerdict::Ok { factor: 8 });
    }

    #[test]
    fn avx512_with_5_regs_per_iter_reduces() {
        // 5 × 8 = 40 > 32 → reduce to 32 / 5 = 6.
        assert_eq!(pick(SimdIsa::Avx512, 5), UnrollVerdict::Ok { factor: 6 });
    }
}
