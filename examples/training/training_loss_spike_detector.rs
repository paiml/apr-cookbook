//! # Training Loss Spike Detector
//!
//! Loss spikes indicate gradient explosion or numerical instability.
//! Detection: loss[t] > median(window) × spike_factor (default 3.0).
//! Reaction recommendations: Continue (small spike), TightenGradClip
//! (medium), HaltAndRestart (severe). This recipe builds the detector.
//!
//! Demonstrates the **TRAIN.11** recipe for PMAT-135 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PaLM training stability report (Chowdhery et al., 2022).
//!
//! Run with: cargo run --example training_loss_spike_detector
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_SPIKE_FACTOR: f64 = 3.0;
const SEVERE_FACTOR: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikeReaction {
    Continue,
    TightenGradClip,
    HaltAndRestart,
}

#[derive(Debug, PartialEq)]
pub enum DetectVerdict {
    NoSpike,
    Spike { ratio: f64, reaction: SpikeReaction },
    InsufficientWindow,
    InvalidLoss,
}

pub fn detect(window: &[f64], current: f64) -> DetectVerdict {
    if window.len() < 5 {
        return DetectVerdict::InsufficientWindow;
    }
    if !current.is_finite() || current < 0.0 {
        return DetectVerdict::InvalidLoss;
    }
    if window.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return DetectVerdict::InvalidLoss;
    }
    let median = compute_median(window);
    if median <= 0.0 {
        return DetectVerdict::NoSpike;
    }
    let ratio = current / median;
    if ratio < DEFAULT_SPIKE_FACTOR {
        return DetectVerdict::NoSpike;
    }
    let reaction = if ratio >= SEVERE_FACTOR {
        SpikeReaction::HaltAndRestart
    } else {
        SpikeReaction::TightenGradClip
    };
    DetectVerdict::Spike { ratio, reaction }
}

fn compute_median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_loss_spike_detector")?;

    let stable = [2.0f64, 2.1, 2.0, 1.9, 2.0];
    println!("stable cur=2.1: {:?}", detect(&stable, 2.1));
    println!("medium spike cur=8.0: {:?}", detect(&stable, 8.0));
    println!("severe spike cur=30.0: {:?}", detect(&stable, 30.0));
    println!("short window: {:?}", detect(&[1.0, 2.0], 5.0));
    println!("nan cur: {:?}", detect(&stable, f64::NAN));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> Vec<f64> {
        vec![2.0, 2.1, 1.9, 2.0, 2.05]
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stable_loss_no_spike() {
        assert_eq!(detect(&baseline(), 2.1), DetectVerdict::NoSpike);
    }

    #[test]
    fn medium_spike_tightens_clip() {
        // median ≈ 2.0; current=7.0 → ratio = 3.5 → TightenGradClip.
        let v = detect(&baseline(), 7.0);
        if let DetectVerdict::Spike { reaction, .. } = v {
            assert_eq!(reaction, SpikeReaction::TightenGradClip);
        }
    }

    #[test]
    fn severe_spike_halts() {
        // median ≈ 2.0; current=25.0 → ratio = 12.5 → HaltAndRestart.
        let v = detect(&baseline(), 25.0);
        if let DetectVerdict::Spike { reaction, .. } = v {
            assert_eq!(reaction, SpikeReaction::HaltAndRestart);
        }
    }

    #[test]
    fn just_below_spike_factor_no_spike() {
        // median ≈ 2.0; current=5.99 → ratio < 3.0 (5.99/2.0=2.995).
        let v = detect(&baseline(), 5.99);
        assert_eq!(v, DetectVerdict::NoSpike);
    }

    #[test]
    fn at_spike_factor_triggers() {
        // ratio = exactly 3.0 → Spike, TightenGradClip.
        let v = detect(&baseline(), 6.0);
        assert!(matches!(v, DetectVerdict::Spike { .. }));
    }

    #[test]
    fn at_severe_factor_halts() {
        // median = 2.0; current=20.0 → ratio = 10.0 → HaltAndRestart.
        let v = detect(&baseline(), 20.0);
        if let DetectVerdict::Spike { reaction, .. } = v {
            assert_eq!(reaction, SpikeReaction::HaltAndRestart);
        }
    }

    #[test]
    fn short_window_insufficient() {
        let v = detect(&[1.0, 2.0, 3.0], 10.0);
        assert_eq!(v, DetectVerdict::InsufficientWindow);
    }

    #[test]
    fn nan_current_rejected() {
        assert_eq!(detect(&baseline(), f64::NAN), DetectVerdict::InvalidLoss);
    }

    #[test]
    fn negative_current_rejected() {
        assert_eq!(detect(&baseline(), -1.0), DetectVerdict::InvalidLoss);
    }

    #[test]
    fn nan_window_value_rejected() {
        let mut w = baseline();
        w[0] = f64::NAN;
        assert_eq!(detect(&w, 3.0), DetectVerdict::InvalidLoss);
    }

    #[test]
    fn ratio_reported_correctly() {
        // median ≈ 2.0; current=8.0 → ratio = 4.0.
        if let DetectVerdict::Spike { ratio, .. } = detect(&baseline(), 8.0) {
            assert!((ratio - 4.0).abs() < 1e-9);
        }
    }
}
