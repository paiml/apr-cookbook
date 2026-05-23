//! # apr rosetta compare-inference — Temperature Modes
//!
//! `apr rosetta compare-inference --temperature <T>` switches between
//! greedy decoding (T=0) and sampling (T>0). Greedy is deterministic and
//! produces bit-identical token sequences when both models agree on
//! argmax; sampling is non-deterministic and requires a seeded RNG +
//! per-token probability comparison instead of token equality. This
//! recipe documents and tests the mode classifier.
//!
//! Demonstrates the **ROSETTA-CMP.3** recipe for PMAT-096 (apr rosetta compare-inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-114 + temperature semantics (T=0 ≡ argmax)
//!
//! Run with: cargo run --example cli_rosetta_compare_inference_temperature_modes
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeMode {
    Greedy,           // T == 0
    LowSampling,      // 0 < T <= 0.7
    StandardSampling, // 0.7 < T <= 1.2
    HighEntropy,      // 1.2 < T <= 2.0
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonStrategy {
    BitIdenticalTokens,
    StatisticalKlDivergence,
}

pub fn classify_mode(temperature: f64) -> DecodeMode {
    if temperature == 0.0 {
        DecodeMode::Greedy
    } else if temperature <= 0.7 {
        DecodeMode::LowSampling
    } else if temperature <= 1.2 {
        DecodeMode::StandardSampling
    } else {
        DecodeMode::HighEntropy
    }
}

pub fn pick_strategy(mode: &DecodeMode) -> ComparisonStrategy {
    match mode {
        DecodeMode::Greedy => ComparisonStrategy::BitIdenticalTokens,
        _ => ComparisonStrategy::StatisticalKlDivergence,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_compare_inference_temperature_modes")?;

    println!("temp    mode                     strategy");
    for t in [0.0_f64, 0.1, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0] {
        let m = classify_mode(t);
        let s = pick_strategy(&m);
        println!("{t:>5.2}  {m:>22?}  {s:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn temperature_zero_is_greedy() {
        // Strict equality, not approximate — 0.0 is the canonical greedy flag.
        assert_eq!(classify_mode(0.0), DecodeMode::Greedy);
    }

    #[test]
    fn greedy_uses_bit_identical_strategy() {
        // Greedy must compare token sequences directly (no statistical noise).
        assert_eq!(
            pick_strategy(&DecodeMode::Greedy),
            ComparisonStrategy::BitIdenticalTokens
        );
    }

    #[test]
    fn sampling_modes_use_kl_divergence() {
        // All non-greedy modes need probabilistic comparison.
        for m in [
            DecodeMode::LowSampling,
            DecodeMode::StandardSampling,
            DecodeMode::HighEntropy,
        ] {
            assert_eq!(
                pick_strategy(&m),
                ComparisonStrategy::StatisticalKlDivergence
            );
        }
    }

    #[test]
    fn boundary_at_0_7_is_low_sampling() {
        assert_eq!(classify_mode(0.7), DecodeMode::LowSampling);
        assert_eq!(classify_mode(0.71), DecodeMode::StandardSampling);
    }

    #[test]
    fn boundary_at_1_2_is_standard() {
        assert_eq!(classify_mode(1.2), DecodeMode::StandardSampling);
        assert_eq!(classify_mode(1.21), DecodeMode::HighEntropy);
    }

    #[test]
    fn high_entropy_distinguishable_from_standard() {
        // Test that the two upper modes are distinct — important for picking
        // the right tolerance band downstream.
        assert_ne!(classify_mode(1.0), classify_mode(1.5));
    }

    #[test]
    fn boundary_at_2_0_is_still_high_entropy() {
        // Upper bound (CLI rejects T > 2.0 in the envelope, but classifier
        // happily accepts the maximum allowed value).
        assert_eq!(classify_mode(2.0), DecodeMode::HighEntropy);
    }
}
