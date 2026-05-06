//! # Advanced KV-Cache Quantization Picker
//!
//! Long contexts blow KV cache budget. Compressing K/V via quantization
//! saves memory but may degrade quality. Tier picker:
//!   short context (< 4k) → fp16 (no compression needed)
//!   medium (4k-32k) → int8 (modest quality loss, 2× compression)
//!   long (32k-128k) → int4 (4× compression; some quality drop)
//!   ultra (≥ 128k) → MixedHotInt8ColdInt4 (recent tokens int8, older int4)
//!
//! Demonstrates the **ADV.16** recipe for PMAT-145 (advanced round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: KV cache quantization (Liu et al. 2024, KIVI).
//!
//! Run with: cargo run --example adv_kv_quantization
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    Fp16,
    Int8,
    Int4,
    MixedHotInt8ColdInt4,
}

#[derive(Debug, PartialEq)]
pub enum QuantVerdict {
    Ok {
        precision: KvPrecision,
        compression_ratio: f64,
        expected_quality_drop_pct: f64,
    },
    InvalidContext,
}

pub fn pick(context_tokens: u32, quality_sensitive: bool) -> QuantVerdict {
    if context_tokens == 0 {
        return QuantVerdict::InvalidContext;
    }
    let precision = if quality_sensitive {
        if context_tokens < 32_768 {
            KvPrecision::Fp16
        } else {
            KvPrecision::Int8
        }
    } else if context_tokens < 4_096 {
        KvPrecision::Fp16
    } else if context_tokens < 32_768 {
        KvPrecision::Int8
    } else if context_tokens < 131_072 {
        KvPrecision::Int4
    } else {
        KvPrecision::MixedHotInt8ColdInt4
    };
    let (compression_ratio, expected_quality_drop_pct) = match precision {
        KvPrecision::Fp16 => (1.0, 0.0),
        KvPrecision::Int8 => (2.0, 0.5),
        KvPrecision::Int4 => (4.0, 2.0),
        KvPrecision::MixedHotInt8ColdInt4 => (3.0, 1.2),
    };
    QuantVerdict::Ok {
        precision,
        compression_ratio,
        expected_quality_drop_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_kv_quantization")?;

    println!("short 1k: {:?}", pick(1024, false));
    println!("medium 16k: {:?}", pick(16_384, false));
    println!("long 64k: {:?}", pick(65_536, false));
    println!("ultra 200k: {:?}", pick(200_000, false));
    println!("quality-sensitive 16k: {:?}", pick(16_384, true));
    println!("invalid: {:?}", pick(0, false));
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
    fn short_context_fp16() {
        let v = pick(1024, false);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Fp16);
        }
    }

    #[test]
    fn medium_context_int8() {
        let v = pick(16_384, false);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Int8);
        }
    }

    #[test]
    fn long_context_int4() {
        let v = pick(65_536, false);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Int4);
        }
    }

    #[test]
    fn ultra_context_mixed() {
        let v = pick(200_000, false);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::MixedHotInt8ColdInt4);
        }
    }

    #[test]
    fn quality_sensitive_avoids_int4() {
        // Long context, but quality_sensitive → Int8 not Int4.
        let v = pick(65_536, true);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Int8);
        }
    }

    #[test]
    fn quality_sensitive_short_fp16() {
        let v = pick(1024, true);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Fp16);
        }
    }

    #[test]
    fn compression_increases_with_quant_level() {
        let fp16 = pick(1024, false);
        let int8 = pick(16_384, false);
        let int4 = pick(65_536, false);
        if let (
            QuantVerdict::Ok {
                compression_ratio: f,
                ..
            },
            QuantVerdict::Ok {
                compression_ratio: i8r,
                ..
            },
            QuantVerdict::Ok {
                compression_ratio: i4r,
                ..
            },
        ) = (fp16, int8, int4)
        {
            assert!(f < i8r);
            assert!(i8r < i4r);
        }
    }

    #[test]
    fn quality_drop_zero_for_fp16() {
        if let QuantVerdict::Ok {
            expected_quality_drop_pct,
            ..
        } = pick(1024, false)
        {
            assert!(expected_quality_drop_pct.abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_context() {
        assert_eq!(pick(0, false), QuantVerdict::InvalidContext);
    }

    #[test]
    fn boundary_at_4k_uses_int8() {
        let v = pick(4_096, false);
        if let QuantVerdict::Ok { precision, .. } = v {
            assert_eq!(precision, KvPrecision::Int8);
        }
    }
}
