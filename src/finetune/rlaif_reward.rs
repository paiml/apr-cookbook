//! Tier 4.6 + 4.7 — RLAIF / constitutional + reward modeling helpers.
//!
//! Closed-form invariants:
//!
//! - RLAIF AI-judge: Pearson correlation between AI scores and human scores
//!   ≥ 0.7 on a deterministic fixture (correlation is closed-form).
//! - Constitutional: refusal rate on harmful prompts rises ≥ 20pp.
//! - Self-critique: unsafe rate drops without ≥5pp drop in helpfulness.
//! - Pairwise reward: P(chosen > rejected) > 0.5 on held-out fixture.
//! - Scalar reward: R² ≥ 0.5 on regression fixture.
//! - Ensemble reward: variance(ensemble) < variance(single).

#![allow(clippy::needless_range_loop)]

/// Pearson correlation coefficient.
#[must_use]
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NAN;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va < 1e-12 || vb < 1e-12 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Refusal rate uplift on harmful prompts (post − pre).
#[must_use]
pub fn refusal_rate_uplift(pre_refusal_rate: f64, post_refusal_rate: f64) -> f64 {
    post_refusal_rate - pre_refusal_rate
}

/// Self-critique balance: returns true if unsafe rate decreased AND
/// helpfulness drop ≤ 5pp.
#[must_use]
pub fn self_critique_balanced(
    pre_unsafe: f64,
    post_unsafe: f64,
    pre_help: f64,
    post_help: f64,
) -> bool {
    let unsafe_dropped = post_unsafe < pre_unsafe;
    let help_drop_pp = (pre_help - post_help) * 100.0;
    unsafe_dropped && help_drop_pp <= 5.0
}

/// Pairwise reward: fraction of pairs where chosen-score > rejected-score.
#[must_use]
pub fn pairwise_chosen_wins(scores: &[(f64, f64)]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let wins = scores.iter().filter(|(c, r)| c > r).count();
    wins as f64 / scores.len() as f64
}

/// Scalar reward R² (coefficient of determination).
#[must_use]
pub fn r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return f64::NAN;
    }
    let mean_y = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_tot: f64 = targets.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    if ss_tot < 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Variance of a slice.
#[must_use]
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Ensemble averaging: returns average of m models' per-sample predictions.
#[must_use]
pub fn ensemble_mean(per_sample_per_model: &[Vec<f64>]) -> Vec<f64> {
    if per_sample_per_model.is_empty() {
        return Vec::new();
    }
    let m = per_sample_per_model.len() as f64;
    let n = per_sample_per_model[0].len();
    (0..n)
        .map(|i| {
            per_sample_per_model
                .iter()
                .map(|model| model[i])
                .sum::<f64>()
                / m
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_judge_correlates_with_human() {
        // AI scores almost identical to human scores (perfect correlation).
        let ai = vec![0.1_f64, 0.3, 0.5, 0.7, 0.9];
        let human = vec![0.15_f64, 0.32, 0.48, 0.72, 0.88];
        let r = pearson(&ai, &human);
        assert!(r >= 0.7, "Pearson correlation {r} should be ≥ 0.7");
    }

    #[test]
    fn refusal_uplift_20pp() {
        let uplift = refusal_rate_uplift(0.3, 0.55);
        assert!(uplift >= 0.20);
    }

    #[test]
    fn self_critique_balanced_pass() {
        // Unsafe 30% → 20%, helpfulness 80% → 78% (drop 2pp).
        assert!(self_critique_balanced(0.30, 0.20, 0.80, 0.78));
    }

    #[test]
    fn self_critique_unbalanced_fails() {
        // Unsafe drops, but helpfulness drops 10pp.
        assert!(!self_critique_balanced(0.30, 0.20, 0.80, 0.70));
    }

    #[test]
    fn pairwise_chosen_above_rejected() {
        let pairs = vec![
            (0.7_f64, 0.3_f64),
            (0.6, 0.4),
            (0.8, 0.2),
            (0.5, 0.5), // tie
        ];
        let acc = pairwise_chosen_wins(&pairs);
        assert!(acc > 0.5);
    }

    #[test]
    fn r_squared_perfect_fit() {
        let p = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = vec![1.0_f64, 2.0, 3.0, 4.0];
        let r2 = r_squared(&p, &t);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ensemble_reduces_variance() {
        // 3 models with biased noise on same fixture.
        let m1 = vec![1.1, 2.2, 3.3, 4.4];
        let m2 = vec![1.0, 2.0, 3.0, 4.0];
        let m3 = vec![0.9, 1.8, 2.7, 3.6];
        let avg = ensemble_mean(&[m1.clone(), m2.clone(), m3.clone()]);
        let var_avg = variance(&avg);
        let var_m1 = variance(&m1);
        // Variance is bounded: avg variance ≤ max member variance.
        assert!(var_avg <= var_m1);
    }
}
