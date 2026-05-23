//! Tier 3.4 Class imbalance — shared helper for 5 recipes.
//!
//! Models the *observable* invariants of common imbalance-handling
//! techniques as closed-form checks.
//!
//! - Weighted CE: inverse-frequency weights make minority-class loss term
//!   strictly larger than uniform weighting.
//! - Focal loss: γ=2 → easy-positive (p=0.9) contribution down-weighted
//!   by (1−p)^γ = 0.01 × baseline = ≥ 90% reduction.
//! - SMOTE: synthetic minority point lies within k-NN convex hull (between
//!   2 minority neighbors).
//! - Threshold tuning: F1 strictly increases vs default 0.5 when class
//!   imbalance shifts the optimal threshold.
//! - Cost-sensitive: asymmetric (10:1) cost matrix shifts the decision
//!   threshold toward the minority class (more positive predictions).

#![allow(clippy::needless_range_loop)]

/// Inverse-frequency class weights for binary classification.
/// w_i = total / (n_classes × class_count_i).
#[must_use]
pub fn inverse_freq_weights(class_counts: &[u32]) -> Vec<f64> {
    let total = class_counts.iter().sum::<u32>() as f64;
    let n_classes = class_counts.len() as f64;
    class_counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                total / (n_classes * f64::from(c))
            }
        })
        .collect()
}

/// Weighted cross-entropy loss for binary fixture given per-sample
/// (probability, label) pairs and per-class weights.
#[must_use]
pub fn weighted_ce(samples: &[(f64, u8)], class_weights: &[f64]) -> f64 {
    if class_weights.len() < 2 {
        return f64::NAN;
    }
    samples
        .iter()
        .map(|(p, y)| {
            let p_clamped = p.clamp(1e-9, 1.0 - 1e-9);
            let nll = if *y == 1 {
                -(p_clamped.ln())
            } else {
                -((1.0 - p_clamped).ln())
            };
            class_weights[*y as usize] * nll
        })
        .sum::<f64>()
        / samples.len() as f64
}

/// Focal-loss factor for one sample: (1 − p_t)^γ × CE.
#[must_use]
pub fn focal_factor(p_correct: f64, gamma: f64) -> f64 {
    (1.0 - p_correct).powf(gamma)
}

/// SMOTE-style synthetic point: for each minority point, walk to its
/// nearest minority neighbor at fraction `t ∈ [0,1]`.
#[must_use]
pub fn smote_synthesize(minority_points: &[(f64, f64)], t: f64) -> Vec<(f64, f64)> {
    if minority_points.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(minority_points.len());
    for (i, &p) in minority_points.iter().enumerate() {
        // Find nearest other minority point.
        let mut best_idx = (i + 1) % minority_points.len();
        let mut best_d = f64::INFINITY;
        for (j, &q) in minority_points.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2);
            if d < best_d {
                best_d = d;
                best_idx = j;
            }
        }
        let q = minority_points[best_idx];
        out.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
    }
    out
}

/// Decision threshold sweep: find threshold maximizing F1 on (prob, label) pairs.
#[must_use]
pub fn best_threshold(samples: &[(f64, u8)]) -> (f64, f64) {
    let mut best_t = 0.5;
    let mut best_f1 = -1.0_f64;
    let candidates: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    for &t in &candidates {
        let f1 = f1_at_threshold(samples, t);
        if f1 > best_f1 {
            best_f1 = f1;
            best_t = t;
        }
    }
    (best_t, best_f1)
}

#[must_use]
pub fn f1_at_threshold(samples: &[(f64, u8)], t: f64) -> f64 {
    let mut tp = 0_u32;
    let mut fp = 0_u32;
    let mut fn_ = 0_u32;
    for (p, y) in samples {
        let pred = u8::from(*p >= t);
        match (pred, *y) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    if tp == 0 {
        return 0.0;
    }
    let precision = f64::from(tp) / f64::from(tp + fp);
    let recall = f64::from(tp) / f64::from(tp + fn_);
    2.0 * precision * recall / (precision + recall)
}

/// Cost-sensitive decision threshold: optimal threshold for asymmetric cost
/// `cost[1, 0]` (false-negative cost) and `cost[0, 1]` (false-positive cost)
/// is t* = cost_fp / (cost_fp + cost_fn).
#[must_use]
pub fn cost_sensitive_threshold(cost_fn: f64, cost_fp: f64) -> f64 {
    cost_fp / (cost_fp + cost_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_freq_weights_inflate_minority() {
        // 90 majority + 10 minority → minority weight 10× larger than majority.
        let w = inverse_freq_weights(&[90, 10]);
        assert!((w[1] - 5.0).abs() < 1e-9, "w[1] = {}", w[1]);
        assert!((w[0] - 0.555_55).abs() < 1e-3, "w[0] = {}", w[0]);
    }

    #[test]
    fn weighted_ce_raises_minority_loss() {
        let samples = vec![(0.5_f64, 1_u8), (0.5, 0)];
        let uniform = weighted_ce(&samples, &[1.0, 1.0]);
        let weighted = weighted_ce(&samples, &[0.5, 5.0]);
        assert!(weighted > uniform);
    }

    #[test]
    fn focal_gamma_2_downweights_easy_positive() {
        // Easy positive: p_correct=0.9 → factor = 0.01 = 1% of baseline (1.0).
        let factor = focal_factor(0.9, 2.0);
        assert!(
            factor < 0.05,
            "easy-positive factor must be < 5%, got {factor}"
        );
    }

    #[test]
    fn smote_at_t05_is_midpoint() {
        let minority = vec![(0.0, 0.0), (1.0, 1.0)];
        let synth = smote_synthesize(&minority, 0.5);
        assert!((synth[0].0 - 0.5).abs() < 1e-12);
        assert!((synth[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn best_threshold_can_exceed_default() {
        // Imbalanced fixture where t=0.3 is better than t=0.5.
        let samples = vec![
            (0.4_f64, 1_u8),
            (0.45, 1),
            (0.55, 0),
            (0.7, 0),
            (0.25, 1),
            (0.8, 0),
        ];
        let (t, f1) = best_threshold(&samples);
        let f1_default = f1_at_threshold(&samples, 0.5);
        assert!(
            f1 >= f1_default,
            "best threshold ({t}, F1={f1}) must be ≥ default (F1={f1_default})"
        );
    }

    #[test]
    fn cost_sensitive_10_to_1_lowers_threshold() {
        // cost_fn = 10, cost_fp = 1 → t* = 1/11 ≈ 0.09 (lower threshold ⇒ more positives).
        let t = cost_sensitive_threshold(10.0, 1.0);
        assert!(t < 0.5, "10:1 cost must lower threshold below 0.5, got {t}");
        assert!((t - 1.0 / 11.0).abs() < 1e-12);
    }
}
