//! # apr quantize — Scheme Size Predictor
//!
//! `apr quantize --scheme <S>` accepts {int8, int4, fp16, q4k}. Each
//! produces a different output size. This recipe builds the predictor
//! as a pure function: input bytes × per-scheme ratio = output bytes.
//! Used by `--plan` mode to estimate the conversion cost before running.
//!
//! Demonstrates the **QUANTIZE.11** recipe for PMAT-105 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-243 + per-scheme bit budgets
//!
//! Run with: cargo run --example cli_quantize_scheme_size_predictor
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Int8,
    Int4,
    Fp16,
    Q4K,
}

impl Scheme {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "int8" => Some(Scheme::Int8),
            "int4" => Some(Scheme::Int4),
            "fp16" => Some(Scheme::Fp16),
            "q4k" => Some(Scheme::Q4K),
            _ => None,
        }
    }

    /// Output bytes per input fp32 byte (1.0 = fp32 baseline).
    pub fn bytes_per_input_byte(self) -> f64 {
        match self {
            Scheme::Fp16 => 0.5,
            Scheme::Int8 => 0.25,
            Scheme::Int4 => 0.125,
            Scheme::Q4K => 0.13, // ~Q4K_M average including scales/mins
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SizePrediction {
    pub scheme: Scheme,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub compression_ratio: f64,
}

pub fn predict(scheme: &str, fp32_input_bytes: u64) -> Option<SizePrediction> {
    let s = Scheme::from_str_strict(scheme)?;
    let output = (fp32_input_bytes as f64 * s.bytes_per_input_byte()) as u64;
    Some(SizePrediction {
        scheme: s,
        input_bytes: fp32_input_bytes,
        output_bytes: output,
        compression_ratio: fp32_input_bytes as f64 / output.max(1) as f64,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_scheme_size_predictor")?;

    let input_bytes = 4_000_000_000u64; // 4 GB fp32 model
    for scheme in ["fp16", "int8", "int4", "q4k", "garbage"] {
        match predict(scheme, input_bytes) {
            Some(p) => println!(
                "{scheme:>8}  in={} GB  out={} GB  compression={:.2}x",
                p.input_bytes / 1_000_000_000,
                p.output_bytes / 1_000_000_000,
                p.compression_ratio
            ),
            None => println!("{scheme:>8}  unknown scheme"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fp16_halves_size() {
        let p = predict("fp16", 1_000_000).unwrap();
        assert_eq!(p.output_bytes, 500_000);
    }

    #[test]
    fn int8_quarters_size() {
        let p = predict("int8", 1_000_000).unwrap();
        assert_eq!(p.output_bytes, 250_000);
    }

    #[test]
    fn int4_eighths_size() {
        let p = predict("int4", 1_000_000).unwrap();
        assert_eq!(p.output_bytes, 125_000);
    }

    #[test]
    fn q4k_close_to_int4() {
        // Q4K_M includes scale/mins overhead → ~0.13 vs int4's 0.125.
        let p = predict("q4k", 1_000_000).unwrap();
        assert!(p.output_bytes >= predict("int4", 1_000_000).unwrap().output_bytes);
    }

    #[test]
    fn unknown_scheme_returns_none() {
        assert!(predict("garbage", 1_000_000).is_none());
        assert!(predict("", 1_000_000).is_none());
    }

    #[test]
    fn compression_ratio_inverse_of_size() {
        // int4 → 8x compression.
        let p = predict("int4", 1_000_000).unwrap();
        assert!((p.compression_ratio - 8.0).abs() < 0.01);
    }

    #[test]
    fn very_large_input_does_not_overflow() {
        let p = predict("int4", 1_000_000_000_000).unwrap();
        assert!(p.output_bytes > 0);
    }
}
