//! # Speech Diarization Speaker-Count Estimator
//!
//! Given speaker-embedding distances, estimate cluster count via
//! "elbow" rule: pick K where decreasing sum-of-squared-distances
//! flattens. Approximation: drop in SSE between K and K+1 falls below
//! 10% of original drop → K is optimal.
//!
//! Demonstrates the **SPEECH.8** recipe for PMAT-149 (speech round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pyannote-audio diarization elbow heuristic.
//!
//! Run with: cargo run --example speech_diarization_count
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CountVerdict {
    Ok { speakers: u32, sse_at_optimal: f64 },
    InsufficientHistory,
    AllSimilar,
    InvalidValues,
}

const ELBOW_DROP_RATIO: f64 = 0.10;

pub fn estimate(sse_per_k: &[f64]) -> CountVerdict {
    if sse_per_k.len() < 2 {
        return CountVerdict::InsufficientHistory;
    }
    if sse_per_k.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return CountVerdict::InvalidValues;
    }
    let initial_drop = sse_per_k[0] - sse_per_k[1];
    if initial_drop.abs() < 1e-9 {
        return CountVerdict::AllSimilar;
    }
    let threshold = initial_drop.abs() * ELBOW_DROP_RATIO;
    for k in 1..sse_per_k.len() - 1 {
        let drop = sse_per_k[k] - sse_per_k[k + 1];
        if drop.abs() < threshold {
            return CountVerdict::Ok {
                speakers: (k + 1) as u32,
                sse_at_optimal: sse_per_k[k],
            };
        }
    }
    let last_idx = sse_per_k.len() - 1;
    CountVerdict::Ok {
        speakers: (last_idx + 1) as u32,
        sse_at_optimal: sse_per_k[last_idx],
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_diarization_count")?;

    // Strong elbow at K=3.
    let sse = [100.0, 50.0, 20.0, 18.0, 17.5, 17.4];
    println!("strong elbow at 3: {:?}", estimate(&sse));

    let no_elbow = [100.0, 90.0, 80.0, 70.0, 60.0];
    println!("gradual no-elbow: {:?}", estimate(&no_elbow));

    let flat = [10.0, 10.0, 10.0];
    println!("flat: {:?}", estimate(&flat));

    println!("insufficient: {:?}", estimate(&[10.0]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn strong_elbow_detected() {
        // K=1 sse=100, K=2 sse=50, K=3 sse=20, then plateau.
        let sse = [100.0, 50.0, 20.0, 18.0, 17.5, 17.4];
        if let CountVerdict::Ok { speakers, .. } = estimate(&sse) {
            assert_eq!(speakers, 3);
        }
    }

    #[test]
    fn gradual_decrease_picks_last() {
        let sse = [100.0, 90.0, 80.0, 70.0, 60.0];
        if let CountVerdict::Ok { speakers, .. } = estimate(&sse) {
            assert_eq!(speakers, 5);
        }
    }

    #[test]
    fn flat_returns_all_similar() {
        let sse = [10.0, 10.0, 10.0];
        assert_eq!(estimate(&sse), CountVerdict::AllSimilar);
    }

    #[test]
    fn insufficient_history_rejected() {
        assert_eq!(estimate(&[10.0]), CountVerdict::InsufficientHistory);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(estimate(&[]), CountVerdict::InsufficientHistory);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(estimate(&[10.0, f64::NAN]), CountVerdict::InvalidValues);
    }

    #[test]
    fn negative_rejected() {
        assert_eq!(estimate(&[10.0, -1.0]), CountVerdict::InvalidValues);
    }

    #[test]
    fn two_points_picks_k2() {
        // Only 2 SSE values: returns K=2 (last).
        let v = estimate(&[100.0, 20.0]);
        if let CountVerdict::Ok { speakers, .. } = v {
            assert_eq!(speakers, 2);
        }
    }

    #[test]
    fn elbow_at_k4() {
        // 100 → 80 → 60 → 40 → 39.5 → 39 (drops 20,20,20,0.5,0.5; elbow at K=4).
        let v = estimate(&[100.0, 80.0, 60.0, 40.0, 39.5, 39.0]);
        if let CountVerdict::Ok { speakers, .. } = v {
            assert_eq!(speakers, 4);
        }
    }

    #[test]
    fn sse_at_optimal_returned() {
        let sse = [100.0, 50.0, 20.0, 18.0, 17.5];
        if let CountVerdict::Ok { sse_at_optimal, .. } = estimate(&sse) {
            assert!((sse_at_optimal - 20.0).abs() < 1e-9);
        }
    }
}
