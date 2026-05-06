//! # Format BF16 vs FP16 Mantissa Bit Explainer
//!
//! Both BF16 and FP16 are 16-bit floats but trade off mantissa for
//! exponent: FP16 has 1 sign / 5 exp / 10 mantissa bits; BF16 has 1
//! sign / 8 exp / 7 mantissa bits. BF16 has FP32's dynamic range (no
//! overflow on large gradients) but coarser precision. This recipe
//! builds the per-format breakdown plus max-representable plus epsilon.
//!
//! Demonstrates the **FMT.18** recipe for PMAT-130 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kalamkar et al. (2019). A Study of BFLOAT16 for Deep Learning Training.
//!
//! Run with: cargo run --example format_bf16_vs_fp16_mantissa
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfFormat {
    Fp16,
    Bf16,
}

impl HalfFormat {
    pub fn sign_bits(self) -> u8 {
        1
    }

    pub fn exponent_bits(self) -> u8 {
        match self {
            HalfFormat::Fp16 => 5,
            HalfFormat::Bf16 => 8,
        }
    }

    pub fn mantissa_bits(self) -> u8 {
        match self {
            HalfFormat::Fp16 => 10,
            HalfFormat::Bf16 => 7,
        }
    }

    pub fn max_exponent_bias(self) -> i32 {
        match self {
            HalfFormat::Fp16 => 15,
            HalfFormat::Bf16 => 127,
        }
    }

    pub fn max_finite(self) -> f64 {
        match self {
            HalfFormat::Fp16 => 65_504.0,
            HalfFormat::Bf16 => 3.39e38,
        }
    }

    pub fn machine_epsilon(self) -> f64 {
        let m = i32::from(self.mantissa_bits());
        2f64.powi(-m)
    }
}

#[derive(Debug, PartialEq)]
pub enum RangeVerdict {
    Representable,
    Overflows { format: HalfFormat },
    UnderflowsToZero { format: HalfFormat },
    InvalidValue,
}

pub fn classify_range(value: f64, format: HalfFormat) -> RangeVerdict {
    if !value.is_finite() {
        return RangeVerdict::InvalidValue;
    }
    let abs = value.abs();
    if abs > format.max_finite() {
        return RangeVerdict::Overflows { format };
    }
    let min_subnormal = match format {
        HalfFormat::Fp16 => 5.96e-8,
        HalfFormat::Bf16 => 1.18e-38,
    };
    if abs > 0.0 && abs < min_subnormal {
        return RangeVerdict::UnderflowsToZero { format };
    }
    RangeVerdict::Representable
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_bf16_vs_fp16_mantissa")?;

    for f in [HalfFormat::Fp16, HalfFormat::Bf16] {
        println!(
            "{f:?}  exp={} mantissa={} max={:e} eps={:e}",
            f.exponent_bits(),
            f.mantissa_bits(),
            f.max_finite(),
            f.machine_epsilon()
        );
    }

    for (v, fmt) in [
        (1.0, HalfFormat::Fp16),
        (1e6, HalfFormat::Fp16),
        (1e6, HalfFormat::Bf16),
        (1e-30, HalfFormat::Fp16),
        (f64::NAN, HalfFormat::Fp16),
    ] {
        println!("{v:e} {fmt:?}  →  {:?}", classify_range(v, fmt));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explainer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fp16_bit_layout_correct() {
        assert_eq!(HalfFormat::Fp16.sign_bits(), 1);
        assert_eq!(HalfFormat::Fp16.exponent_bits(), 5);
        assert_eq!(HalfFormat::Fp16.mantissa_bits(), 10);
    }

    #[test]
    fn bf16_bit_layout_correct() {
        assert_eq!(HalfFormat::Bf16.sign_bits(), 1);
        assert_eq!(HalfFormat::Bf16.exponent_bits(), 8);
        assert_eq!(HalfFormat::Bf16.mantissa_bits(), 7);
    }

    #[test]
    fn bit_total_is_16_for_both() {
        for f in [HalfFormat::Fp16, HalfFormat::Bf16] {
            assert_eq!(f.sign_bits() + f.exponent_bits() + f.mantissa_bits(), 16);
        }
    }

    #[test]
    fn bf16_max_far_larger_than_fp16() {
        assert!(HalfFormat::Bf16.max_finite() > HalfFormat::Fp16.max_finite() * 1e30);
    }

    #[test]
    fn fp16_epsilon_smaller_than_bf16() {
        // More mantissa bits → finer precision → smaller epsilon.
        assert!(HalfFormat::Fp16.machine_epsilon() < HalfFormat::Bf16.machine_epsilon());
    }

    #[test]
    fn typical_value_representable_in_both() {
        assert_eq!(
            classify_range(1.0, HalfFormat::Fp16),
            RangeVerdict::Representable
        );
        assert_eq!(
            classify_range(1.0, HalfFormat::Bf16),
            RangeVerdict::Representable
        );
    }

    #[test]
    fn fp16_overflows_at_1e6() {
        let v = classify_range(1e6, HalfFormat::Fp16);
        assert!(matches!(v, RangeVerdict::Overflows { .. }));
    }

    #[test]
    fn bf16_handles_1e6_easily() {
        assert_eq!(
            classify_range(1e6, HalfFormat::Bf16),
            RangeVerdict::Representable
        );
    }

    #[test]
    fn fp16_underflows_at_tiny_subnormal() {
        let v = classify_range(1e-30, HalfFormat::Fp16);
        assert!(matches!(v, RangeVerdict::UnderflowsToZero { .. }));
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            classify_range(f64::NAN, HalfFormat::Fp16),
            RangeVerdict::InvalidValue
        );
    }
}
