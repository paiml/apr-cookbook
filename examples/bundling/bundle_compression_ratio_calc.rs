//! # Bundle Compression Ratio Calculator
//!
//! Compression ratio = uncompressed / compressed. Tiers: < 1.5x =
//! noise (compression overhead exceeds savings); 1.5-3x = typical for
//! quantized tensors; 3-10x = good for FP32 weights with LZ4; > 10x =
//! suspicious (likely all-zero or malformed). This recipe builds the
//! calculator + savings classifier.
//!
//! Demonstrates the **BUNDLE.12** recipe for PMAT-127 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Collet, Y. (2013). LZ4 Block Format Specification.
//!
//! Run with: cargo run --example bundle_compression_ratio_calc
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RatioTier {
    Noise,
    Typical,
    Good,
    Suspect,
    InvalidBytes,
}

pub fn ratio(uncompressed_bytes: u64, compressed_bytes: u64) -> Option<f64> {
    if uncompressed_bytes == 0 || compressed_bytes == 0 {
        return None;
    }
    Some(uncompressed_bytes as f64 / compressed_bytes as f64)
}

pub fn classify(uncompressed_bytes: u64, compressed_bytes: u64) -> RatioTier {
    let Some(r) = ratio(uncompressed_bytes, compressed_bytes) else {
        return RatioTier::InvalidBytes;
    };
    if r < 1.5 {
        RatioTier::Noise
    } else if r < 3.0 {
        RatioTier::Typical
    } else if r < 10.0 {
        RatioTier::Good
    } else {
        RatioTier::Suspect
    }
}

pub fn savings_pct(uncompressed_bytes: u64, compressed_bytes: u64) -> Option<f64> {
    let r = ratio(uncompressed_bytes, compressed_bytes)?;
    Some((1.0 - 1.0 / r) * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_compression_ratio_calc")?;

    let cases = [
        (1024u64, 800u64),
        (1024, 500),
        (1024, 256),
        (1024, 50),
        (1024, 0),
    ];
    for (u, c) in cases {
        println!(
            "{}/{}  ratio={:?}  tier={:?}  savings={:?}%",
            u,
            c,
            ratio(u, c),
            classify(u, c),
            savings_pct(u, c)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ratio_basic_math() {
        assert_eq!(ratio(1000, 250), Some(4.0));
    }

    #[test]
    fn zero_inputs_yield_none() {
        assert!(ratio(0, 100).is_none());
        assert!(ratio(100, 0).is_none());
    }

    #[test]
    fn under_1_5x_noise() {
        // 1024 / 800 = 1.28.
        assert_eq!(classify(1024, 800), RatioTier::Noise);
    }

    #[test]
    fn one_point_5_to_3x_typical() {
        // 1024 / 500 = 2.05.
        assert_eq!(classify(1024, 500), RatioTier::Typical);
    }

    #[test]
    fn three_to_10x_good() {
        // 1024 / 256 = 4.0.
        assert_eq!(classify(1024, 256), RatioTier::Good);
    }

    #[test]
    fn over_10x_suspect() {
        // 1024 / 50 ≈ 20.5.
        assert_eq!(classify(1024, 50), RatioTier::Suspect);
    }

    #[test]
    fn invalid_bytes_classified() {
        assert_eq!(classify(0, 100), RatioTier::InvalidBytes);
    }

    #[test]
    fn savings_pct_basic_math() {
        // 4× ratio → 75% savings.
        let pct = savings_pct(1000, 250).unwrap();
        assert!((pct - 75.0).abs() < 1e-9);
    }

    #[test]
    fn savings_pct_high_compression() {
        // 10× → 90%.
        let pct = savings_pct(1000, 100).unwrap();
        assert!((pct - 90.0).abs() < 1e-9);
    }

    #[test]
    fn boundary_at_1_5x_typical() {
        // 1500 / 1000 = 1.5 — boundary inclusive (>= 1.5 → Typical).
        assert_eq!(classify(1500, 1000), RatioTier::Typical);
    }
}
