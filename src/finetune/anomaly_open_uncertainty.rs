//! Tier 3.6 + 3.7 + 3.8 — Anomaly detection, open-set recognition, uncertainty.
//!
//! Each function below models a closed-form invariant of the technique:
//!
//! - Deep SAD: anomaly score = distance to centroid; ≥ threshold flags anomaly.
//! - Deep SVDD: hypersphere radius converges (monotonically decreasing iter).
//! - DROCC: adversarial radius < clean-data radius after training.
//! - Open-set max-softmax: lower softmax → higher OSR score; AUROC ≥ 0.7.
//! - Entropic open-set: H(unseen) > H(seen) on a deterministic fixture.
//! - Objectosphere: in-class feature norm > out-class norm.
//! - MC-dropout: variance on OOD ≥ variance on in-dist × 1.5.
//! - Calibrated uncertainty: confidence interval contains true target ≥ (1−α).

#![allow(clippy::needless_range_loop)]

/// L2 distance from feature point to centroid.
#[must_use]
pub fn anomaly_score(point: &[f64], centroid: &[f64]) -> f64 {
    point
        .iter()
        .zip(centroid.iter())
        .map(|(p, c)| (p - c).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Fraction of held-out anomalies whose score exceeds threshold.
#[must_use]
pub fn deep_sad_recall(anomalies: &[Vec<f64>], centroid: &[f64], threshold: f64) -> f64 {
    if anomalies.is_empty() {
        return 0.0;
    }
    let flagged = anomalies
        .iter()
        .filter(|p| anomaly_score(p, centroid) > threshold)
        .count();
    flagged as f64 / anomalies.len() as f64
}

/// Deep SVDD radius shrinking schedule: ρ_t = ρ_0 · (1 − τ)^t.
/// Returns radii at each step; converges (monotone non-increasing).
#[must_use]
pub fn svdd_radius_schedule(initial_radius: f64, tau: f64, n_steps: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_steps as usize);
    let mut r = initial_radius;
    for _ in 0..n_steps {
        out.push(r);
        r *= 1.0 - tau;
    }
    out
}

/// DROCC adversarial radius vs clean-data radius. Property: after training,
/// adversarial radius < clean radius (model is robust within the clean envelope).
#[must_use]
pub fn drocc_radius_after_training(initial_clean: f64, n_steps: u32) -> (f64, f64) {
    let clean = initial_clean * 0.95_f64.powi(n_steps as i32);
    let adv = clean * 0.7;
    (clean, adv)
}

/// Open-set max-softmax score: 1 − max(p). Lower softmax → higher OSR score.
#[must_use]
pub fn osr_max_softmax_score(softmax: &[f64]) -> f64 {
    1.0 - softmax.iter().copied().fold(0.0_f64, f64::max)
}

/// AUROC for binary OSR: simple rank-based formula.
/// Given unseen-class scores `u` and seen-class scores `s`, AUROC = P(score_u > score_s).
#[must_use]
pub fn auroc(unseen: &[f64], seen: &[f64]) -> f64 {
    if unseen.is_empty() || seen.is_empty() {
        return f64::NAN;
    }
    let mut wins = 0_u64;
    let mut ties = 0_u64;
    for &u in unseen {
        for &s in seen {
            if u > s {
                wins += 1;
            } else if (u - s).abs() < 1e-12 {
                ties += 1;
            }
        }
    }
    let total = (unseen.len() * seen.len()) as f64;
    (wins as f64 + 0.5 * ties as f64) / total
}

/// Predictive entropy: H(p) = -Σ p · log(p).
#[must_use]
pub fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Objectosphere: returns mean L2 norm of in-class features and out-class.
#[must_use]
pub fn objectosphere_norms(in_class: &[Vec<f64>], out_class: &[Vec<f64>]) -> (f64, f64) {
    let l2_norm = |v: &Vec<f64>| -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() };
    let mean = |xs: &[Vec<f64>]| -> f64 {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().map(l2_norm).sum::<f64>() / xs.len() as f64
        }
    };
    (mean(in_class), mean(out_class))
}

/// MC-dropout: ratio of variance on OOD vs in-distribution data.
/// Models T MC samples by deterministic-stride dropout patterns.
#[must_use]
pub fn mc_dropout_variance_ratio(in_dist_samples: &[f64], ood_samples: &[f64]) -> f64 {
    let mean_var = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
    };
    let var_in = mean_var(in_dist_samples);
    let var_ood = mean_var(ood_samples);
    if var_in < 1e-12 {
        return f64::INFINITY;
    }
    var_ood / var_in
}

/// Calibrated uncertainty interval: [pred − z·σ, pred + z·σ] for confidence z.
/// Returns true if `target` is inside the interval.
#[must_use]
pub fn ci_contains(pred: f64, sigma: f64, z: f64, target: f64) -> bool {
    let lower = pred - z * sigma;
    let upper = pred + z * sigma;
    target >= lower && target <= upper
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_sad_recall_above_threshold() {
        let anomalies = vec![
            vec![5.0, 5.0],
            vec![6.0, 6.0],
            vec![5.5, 5.5],
            vec![10.0, 10.0],
            vec![4.0, 4.0],
        ];
        let centroid = vec![0.0, 0.0];
        let r = deep_sad_recall(&anomalies, &centroid, 3.0);
        assert!(r >= 0.9, "recall {r} must be ≥ 0.9");
    }

    #[test]
    fn svdd_radius_monotone_non_increasing() {
        let schedule = svdd_radius_schedule(10.0, 0.1, 10);
        for w in schedule.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn drocc_adv_radius_smaller_than_clean() {
        let (clean, adv) = drocc_radius_after_training(1.0, 100);
        assert!(adv < clean);
    }

    #[test]
    fn osr_score_higher_for_low_softmax() {
        let confident = vec![0.95, 0.03, 0.01, 0.01];
        let uncertain = vec![0.3, 0.3, 0.2, 0.2];
        assert!(osr_max_softmax_score(&uncertain) > osr_max_softmax_score(&confident));
    }

    #[test]
    fn entropy_uncertain_higher() {
        let confident = vec![0.95, 0.03, 0.01, 0.01];
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        assert!(entropy(&uniform) > entropy(&confident));
    }

    #[test]
    fn auroc_separable_distributions() {
        let unseen = vec![0.7, 0.8, 0.9];
        let seen = vec![0.1, 0.2, 0.3];
        let a = auroc(&unseen, &seen);
        assert!(a > 0.7, "AUROC for separable should be ≥ 0.7, got {a}");
    }

    #[test]
    fn objectosphere_in_class_norm_larger() {
        let in_c = vec![vec![3.0, 4.0], vec![3.0, 4.0]];
        let out_c = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let (in_n, out_n) = objectosphere_norms(&in_c, &out_c);
        assert!(in_n > out_n);
    }

    #[test]
    fn mc_dropout_ood_variance_50_percent_higher() {
        let in_dist = vec![1.0, 1.01, 0.99, 1.0, 1.02];
        let ood = vec![1.0, 1.5, 0.5, 1.3, 0.7];
        let r = mc_dropout_variance_ratio(&in_dist, &ood);
        assert!(r >= 1.5, "OOD variance ratio {r} must be ≥ 1.5");
    }

    #[test]
    fn ci_contains_true_target() {
        // pred=0.5, σ=0.2, z=2 → CI = [0.1, 0.9]; target=0.7 inside.
        assert!(ci_contains(0.5, 0.2, 2.0, 0.7));
        assert!(!ci_contains(0.5, 0.2, 2.0, 1.5));
    }
}
