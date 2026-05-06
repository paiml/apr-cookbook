//! # apr quantize — `--format` Compatibility Matrix
//!
//! `apr quantize --scheme <S> --format <F>` accepts (scheme, format)
//! pairs. Not every scheme is writable to every format: int4 only fits
//! in GGUF (with k-quant) or APR; fp16 fits anywhere; q4k is GGUF-only.
//! This recipe builds the compatibility matrix.
//!
//! Demonstrates the **QUANTIZE.12** recipe for PMAT-105 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-243 + per-format quant support
//!
//! Run with: cargo run --example cli_quantize_format_compatibility
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Fp16,
    Int8,
    Int4,
    Q4K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Apr,
    Gguf,
    SafeTensors,
}

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Ok,
    SchemeNotSupportedByFormat { scheme: Scheme, format: Format },
}

pub fn check(scheme: Scheme, format: Format) -> CompatVerdict {
    let ok = match (scheme, format) {
        (Scheme::Fp16, _) => true,
        (Scheme::Int8, Format::Apr | Format::SafeTensors) => true,
        (Scheme::Int8, Format::Gguf) => true,
        (Scheme::Int4, Format::Apr | Format::Gguf) => true,
        (Scheme::Int4, Format::SafeTensors) => false,
        (Scheme::Q4K, Format::Gguf | Format::Apr) => true,
        (Scheme::Q4K, Format::SafeTensors) => false,
    };
    if ok {
        CompatVerdict::Ok
    } else {
        CompatVerdict::SchemeNotSupportedByFormat { scheme, format }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_format_compatibility")?;

    let schemes = [Scheme::Fp16, Scheme::Int8, Scheme::Int4, Scheme::Q4K];
    let formats = [Format::Apr, Format::Gguf, Format::SafeTensors];

    println!("Scheme   APR  GGUF  ST");
    for s in schemes {
        let row: Vec<&str> = formats
            .iter()
            .map(|f| {
                if check(s, *f) == CompatVerdict::Ok {
                    "✓"
                } else {
                    "✗"
                }
            })
            .collect();

        println!("{:>6?}   {}   {}   {}", s, row[0], row[1], row[2]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fp16_supported_everywhere() {
        for f in [Format::Apr, Format::Gguf, Format::SafeTensors] {
            assert_eq!(check(Scheme::Fp16, f), CompatVerdict::Ok);
        }
    }

    #[test]
    fn int4_not_supported_by_safetensors() {
        // SafeTensors lacks native int4 dtype; would silently lose precision.
        let v = check(Scheme::Int4, Format::SafeTensors);
        assert!(matches!(
            v,
            CompatVerdict::SchemeNotSupportedByFormat { .. }
        ));
    }

    #[test]
    fn int4_supported_by_apr_and_gguf() {
        assert_eq!(check(Scheme::Int4, Format::Apr), CompatVerdict::Ok);
        assert_eq!(check(Scheme::Int4, Format::Gguf), CompatVerdict::Ok);
    }

    #[test]
    fn q4k_only_for_apr_and_gguf() {
        assert_eq!(check(Scheme::Q4K, Format::Apr), CompatVerdict::Ok);
        assert_eq!(check(Scheme::Q4K, Format::Gguf), CompatVerdict::Ok);
        assert!(matches!(
            check(Scheme::Q4K, Format::SafeTensors),
            CompatVerdict::SchemeNotSupportedByFormat { .. }
        ));
    }

    #[test]
    fn int8_supported_by_all_three() {
        for f in [Format::Apr, Format::Gguf, Format::SafeTensors] {
            assert_eq!(check(Scheme::Int8, f), CompatVerdict::Ok);
        }
    }
}
