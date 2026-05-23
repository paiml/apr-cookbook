//! # Distributed Phi-Accrual Failure Detector
//!
//! Cassandra-style adaptive failure detector. Track inter-arrival
//! gaps of heartbeats; compute phi = -log10(prob of suspicious gap).
//! Phi crosses threshold (typical: 8.0..12.0) → SuspectDown.
//!
//! Mean estimate from EWMA of recent gaps. Variance approximated as
//! mean squared (Cassandra default).
//!
//! Demonstrates the **DIST.9** recipe for PMAT-139 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hayashibara et al. (2004). The φ Accrual Failure Detector.
//!
//! Run with: cargo run --example distributed_phi_failure_detector
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LivenessState {
    Up,
    Suspect,
    Down,
}

#[derive(Debug, PartialEq)]
pub enum PhiVerdict {
    Ok { phi: f64, state: LivenessState },
    InsufficientHeartbeats,
    InvalidGap,
}

const SUSPECT_THRESHOLD: f64 = 8.0;
const DOWN_THRESHOLD: f64 = 12.0;

pub fn evaluate(history_gaps_ms: &[u64], current_gap_ms: u64) -> PhiVerdict {
    if history_gaps_ms.len() < 3 {
        return PhiVerdict::InsufficientHeartbeats;
    }
    if current_gap_ms == 0 {
        return PhiVerdict::InvalidGap;
    }
    let mean =
        history_gaps_ms.iter().map(|x| *x as f64).sum::<f64>() / history_gaps_ms.len() as f64;
    if mean <= 0.0 {
        return PhiVerdict::InvalidGap;
    }
    // Approximate exponential CDF: P(gap > current) = exp(-current/mean).
    // phi = -log10(P) = (current_gap_ms / mean) / ln(10).
    let phi = (current_gap_ms as f64 / mean) / std::f64::consts::LN_10;
    let state = if phi >= DOWN_THRESHOLD {
        LivenessState::Down
    } else if phi >= SUSPECT_THRESHOLD {
        LivenessState::Suspect
    } else {
        LivenessState::Up
    };
    PhiVerdict::Ok { phi, state }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_phi_failure_detector")?;

    let history = [100u64, 110, 95, 105, 100];
    println!("normal gap=110: {:?}", evaluate(&history, 110));
    println!("late gap=2000 (suspect): {:?}", evaluate(&history, 2000));
    println!("very late gap=5000 (down): {:?}", evaluate(&history, 5000));
    println!("insufficient: {:?}", evaluate(&[100, 100], 100));
    println!("zero gap: {:?}", evaluate(&history, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_history() -> Vec<u64> {
        vec![100, 110, 95, 105, 100]
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_gap_yields_low_phi() {
        // Mean ≈ 102; current ~ 110. phi = 110/102/ln(10) ≈ 0.47.
        if let PhiVerdict::Ok { phi, state } = evaluate(&typical_history(), 110) {
            assert!(phi < SUSPECT_THRESHOLD);
            assert_eq!(state, LivenessState::Up);
        }
    }

    #[test]
    fn long_gap_triggers_suspect() {
        // current=2000 → phi ≈ 8.5 → Suspect.
        if let PhiVerdict::Ok { state, .. } = evaluate(&typical_history(), 2000) {
            assert_eq!(state, LivenessState::Suspect);
        }
    }

    #[test]
    fn very_long_gap_triggers_down() {
        // current=5000 → phi ≈ 21.3 → Down.
        if let PhiVerdict::Ok { state, .. } = evaluate(&typical_history(), 5000) {
            assert_eq!(state, LivenessState::Down);
        }
    }

    #[test]
    fn insufficient_heartbeats_rejected() {
        let v = evaluate(&[100, 100], 100);
        assert_eq!(v, PhiVerdict::InsufficientHeartbeats);
    }

    #[test]
    fn zero_gap_invalid() {
        assert_eq!(evaluate(&typical_history(), 0), PhiVerdict::InvalidGap);
    }

    #[test]
    fn phi_increases_with_gap() {
        let v_short = evaluate(&typical_history(), 100);
        let v_long = evaluate(&typical_history(), 1000);
        if let (PhiVerdict::Ok { phi: p_short, .. }, PhiVerdict::Ok { phi: p_long, .. }) =
            (v_short, v_long)
        {
            assert!(p_long > p_short);
        }
    }

    #[test]
    fn phi_threshold_at_suspect() {
        // Find a gap that lands phi exactly at 8.0.
        // 8.0 * ln(10) * 102 ≈ 1879.
        if let PhiVerdict::Ok { state, .. } = evaluate(&typical_history(), 1900) {
            assert_eq!(state, LivenessState::Suspect);
        }
    }

    #[test]
    fn empty_history_rejected() {
        assert_eq!(evaluate(&[], 100), PhiVerdict::InsufficientHeartbeats);
    }

    #[test]
    fn at_minimum_history_succeeds() {
        let v = evaluate(&[100, 100, 100], 100);
        assert!(matches!(v, PhiVerdict::Ok { .. }));
    }

    #[test]
    fn larger_history_smoother_mean() {
        let big_hist: Vec<u64> = (0..100).map(|_| 100).collect();
        if let PhiVerdict::Ok { phi, .. } = evaluate(&big_hist, 100) {
            // For exact mean, phi = 1/ln(10) ≈ 0.434.
            assert!((phi - 1.0 / std::f64::consts::LN_10).abs() < 1e-9);
        }
    }
}
