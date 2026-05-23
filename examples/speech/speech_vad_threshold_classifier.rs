//! # Speech VAD Threshold Classifier
//!
//! Voice Activity Detection (VAD) tags audio frames as speech vs.
//! silence by comparing per-frame RMS energy against a threshold. Too
//! low: false positives; too high: clipped speech. Recommended
//! envelope: −60 dBFS to −20 dBFS. This recipe builds the classifier +
//! adaptive-threshold heuristic.
//!
//! Demonstrates the **SPEECH.3** recipe for PMAT-123 (speech coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sohn et al. (1999). A statistical model-based voice activity detection. IEEE SPL 6(1).
//!
//! Run with: cargo run --example speech_vad_threshold_classifier
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum VadVerdict {
    Speech,
    Silence,
    InvalidEnergy,
}

const MIN_DB: f64 = -60.0;
const MAX_DB: f64 = -20.0;

pub fn classify(frame_rms_dbfs: f64, threshold_dbfs: f64) -> VadVerdict {
    if !frame_rms_dbfs.is_finite() || !threshold_dbfs.is_finite() {
        return VadVerdict::InvalidEnergy;
    }
    if !(MIN_DB..=MAX_DB + 5.0).contains(&threshold_dbfs) {
        return VadVerdict::InvalidEnergy;
    }
    if frame_rms_dbfs >= threshold_dbfs {
        VadVerdict::Speech
    } else {
        VadVerdict::Silence
    }
}

pub fn adaptive_threshold(noise_floor_dbfs: f64, headroom_db: f64) -> Option<f64> {
    if !noise_floor_dbfs.is_finite() || !headroom_db.is_finite() {
        return None;
    }
    if headroom_db <= 0.0 || headroom_db > 30.0 {
        return None;
    }
    let proposed = noise_floor_dbfs + headroom_db;
    Some(proposed.clamp(MIN_DB, MAX_DB))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_vad_threshold_classifier")?;

    for (rms, thresh) in [(-30.0, -40.0), (-50.0, -40.0), (-25.0, -25.0)] {
        println!("rms={rms} thresh={thresh}  →  {:?}", classify(rms, thresh));
    }
    for (floor, head) in [(-55.0, 10.0), (-40.0, 6.0), (-50.0, 50.0)] {
        println!(
            "floor={floor} headroom={head}  →  {:?}",
            adaptive_threshold(floor, head)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_rms_above_threshold_speech() {
        assert_eq!(classify(-30.0, -40.0), VadVerdict::Speech);
    }

    #[test]
    fn low_rms_below_threshold_silence() {
        assert_eq!(classify(-50.0, -40.0), VadVerdict::Silence);
    }

    #[test]
    fn at_threshold_treated_as_speech() {
        // ≥ threshold inclusive — avoid edge flicker.
        assert_eq!(classify(-40.0, -40.0), VadVerdict::Speech);
    }

    #[test]
    fn nan_energy_invalid() {
        assert_eq!(classify(f64::NAN, -40.0), VadVerdict::InvalidEnergy);
        assert_eq!(classify(-30.0, f64::NAN), VadVerdict::InvalidEnergy);
    }

    #[test]
    fn out_of_range_threshold_invalid() {
        // > MAX_DB + headroom.
        assert_eq!(classify(-30.0, 0.0), VadVerdict::InvalidEnergy);
        // < MIN_DB.
        assert_eq!(classify(-30.0, -100.0), VadVerdict::InvalidEnergy);
    }

    #[test]
    fn adaptive_typical_returns_clamped_value() {
        // Floor=-55, head=10 → -45.
        assert!((adaptive_threshold(-55.0, 10.0).unwrap() - (-45.0)).abs() < 1e-9);
    }

    #[test]
    fn adaptive_clamps_to_max_db() {
        // Floor=-25, head=20 → -5, clamps to MAX_DB.
        let t = adaptive_threshold(-25.0, 20.0).unwrap();
        assert!((t - MAX_DB).abs() < 1e-9);
    }

    #[test]
    fn adaptive_clamps_to_min_db() {
        // Floor=-100, head=5 → -95, clamps to MIN_DB.
        let t = adaptive_threshold(-100.0, 5.0).unwrap();
        assert!((t - MIN_DB).abs() < 1e-9);
    }

    #[test]
    fn adaptive_zero_or_negative_headroom_rejected() {
        assert!(adaptive_threshold(-50.0, 0.0).is_none());
        assert!(adaptive_threshold(-50.0, -5.0).is_none());
    }

    #[test]
    fn adaptive_excessive_headroom_rejected() {
        // Headroom > 30 dB is unrealistic and likely a mis-config.
        assert!(adaptive_threshold(-50.0, 50.0).is_none());
    }
}
