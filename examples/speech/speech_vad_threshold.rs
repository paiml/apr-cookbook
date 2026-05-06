//! # Speech VAD Threshold Picker
//!
//! Voice-Activity-Detection thresholds vary by SNR:
//!   high SNR (≥ 20 dB) → low energy threshold + low ZCR threshold
//!   medium SNR (10-20) → balanced
//!   low SNR (< 10 dB) → high thresholds; many false negatives
//!
//! Returns (energy_threshold, zcr_threshold) tuned for the SNR.
//!
//! Demonstrates the **SPEECH.7** recipe for PMAT-149 (speech round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Silero VAD + classic energy/ZCR algorithm.
//!
//! Run with: cargo run --example speech_vad_threshold
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnrTier {
    HighClean,
    Moderate,
    LowNoisy,
}

#[derive(Debug, PartialEq)]
pub enum VadVerdict {
    Ok {
        energy_threshold: f64,
        zcr_threshold: f64,
        tier: SnrTier,
    },
    InvalidSnr,
}

pub fn pick(snr_db: f64) -> VadVerdict {
    if !snr_db.is_finite() {
        return VadVerdict::InvalidSnr;
    }
    let (energy_threshold, zcr_threshold, tier) = if snr_db >= 20.0 {
        (0.01, 0.05, SnrTier::HighClean)
    } else if snr_db >= 10.0 {
        (0.03, 0.10, SnrTier::Moderate)
    } else {
        (0.10, 0.20, SnrTier::LowNoisy)
    };
    VadVerdict::Ok {
        energy_threshold,
        zcr_threshold,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_vad_threshold")?;

    println!("clean (30 dB): {:?}", pick(30.0));
    println!("medium (15 dB): {:?}", pick(15.0));
    println!("noisy (5 dB): {:?}", pick(5.0));
    println!("invalid: {:?}", pick(f64::NAN));
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
    fn high_snr_clean_tier() {
        let v = pick(30.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::HighClean);
        }
    }

    #[test]
    fn medium_snr_moderate_tier() {
        let v = pick(15.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::Moderate);
        }
    }

    #[test]
    fn low_snr_noisy_tier() {
        let v = pick(5.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::LowNoisy);
        }
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(pick(f64::NAN), VadVerdict::InvalidSnr);
    }

    #[test]
    fn higher_snr_lower_thresholds() {
        let high = pick(30.0);
        let low = pick(5.0);
        if let (
            VadVerdict::Ok {
                energy_threshold: h,
                ..
            },
            VadVerdict::Ok {
                energy_threshold: l,
                ..
            },
        ) = (high, low)
        {
            assert!(h < l);
        }
    }

    #[test]
    fn boundary_at_20_db_clean() {
        let v = pick(20.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::HighClean);
        }
    }

    #[test]
    fn boundary_at_10_db_moderate() {
        let v = pick(10.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::Moderate);
        }
    }

    #[test]
    fn just_below_10_db_noisy() {
        let v = pick(9.99);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::LowNoisy);
        }
    }

    #[test]
    fn negative_snr_noisy() {
        // Negative SNR (noise > signal) → noisy tier.
        let v = pick(-5.0);
        if let VadVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SnrTier::LowNoisy);
        }
    }

    #[test]
    fn zcr_threshold_higher_for_noisy() {
        let high = pick(30.0);
        let low = pick(5.0);
        if let (
            VadVerdict::Ok {
                zcr_threshold: h, ..
            },
            VadVerdict::Ok {
                zcr_threshold: l, ..
            },
        ) = (high, low)
        {
            assert!(l > h);
        }
    }
}
