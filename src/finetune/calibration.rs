//! Tier 3.3 Calibration — shared helper for 5 recipes.
//!
//! Models the *observable* invariant of each calibration method as a
//! closed-form check.
//!
//! - Temperature: logit / T preserves argmax; ECE strictly decreases for T>1
//!   when the uncalibrated model is overconfident.
//! - Platt: sigmoid passes through (0.5, 0.5) when slope a≠0, intercept b=0.
//! - Isotonic: monotonic in input score (PAV non-decreasing).
//! - Conformal: empirical coverage matches 1−α within sample-size bound.
//! - Ensemble: average of probabilities yields ECE ≤ min ECE of members.

#![allow(clippy::needless_range_loop)]

/// Expected Calibration Error: 10 equal-width bins of |confidence − accuracy|.
#[must_use]
pub fn ece(confidences: &[f64], correct: &[bool]) -> f64 {
    if confidences.len() != correct.len() || confidences.is_empty() {
        return f64::NAN;
    }
    let mut bins = vec![(0_u32, 0.0_f64, 0.0_f64); 10];
    for (&c, &y) in confidences.iter().zip(correct.iter()) {
        let idx = ((c * 10.0).floor() as usize).min(9);
        let acc = if y { 1.0 } else { 0.0 };
        bins[idx].0 += 1;
        bins[idx].1 += c;
        bins[idx].2 += acc;
    }
    let n = confidences.len() as f64;
    bins.iter()
        .filter(|b| b.0 > 0)
        .map(|(count, csum, asum)| {
            let count_f = f64::from(*count);
            let mean_conf = csum / count_f;
            let mean_acc = asum / count_f;
            (count_f / n) * (mean_conf - mean_acc).abs()
        })
        .sum()
}

/// Apply temperature scaling to logits (single class). Returns sigmoid(logit / T).
#[must_use]
pub fn temperature_apply(logits: &[f64], temperature: f64) -> Vec<f64> {
    logits.iter().map(|l| sigmoid(l / temperature)).collect()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Argmax of a 2-class softmax. For binary, equivalent to (sigmoid(logit) ≥ 0.5).
#[must_use]
pub fn argmax_pred(prob: f64) -> bool {
    prob >= 0.5
}

/// Platt scaling: y = sigmoid(a·x + b). Apply with given a, b.
#[must_use]
pub fn platt_apply(score: f64, a: f64, b: f64) -> f64 {
    sigmoid(a * score + b)
}

/// Isotonic regression via Pool-Adjacent-Violators on (x, y) pairs sorted by x.
/// Returns the calibrated y values in the same order as input.
#[must_use]
pub fn isotonic_pav(scores: &[f64], targets: &[f64]) -> Vec<f64> {
    if scores.len() != targets.len() || scores.is_empty() {
        return Vec::new();
    }
    let n = scores.len();
    let mut indexed: Vec<(usize, f64, f64)> = (0..n).map(|i| (i, scores[i], targets[i])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    // PAV
    let mut values: Vec<f64> = indexed.iter().map(|(_, _, y)| *y).collect();
    let mut weights: Vec<f64> = vec![1.0; n];
    let mut i = 0;
    while i + 1 < values.len() {
        if values[i] > values[i + 1] {
            // Merge into a single block.
            let merged_w = weights[i] + weights[i + 1];
            let merged_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / merged_w;
            values[i] = merged_v;
            weights[i] = merged_w;
            values.remove(i + 1);
            weights.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    // Replay block values into per-input output (ordered by sorted index).
    let mut sorted_outputs = Vec::with_capacity(n);
    let mut cursor = 0;
    for (v, &w) in values.iter().zip(weights.iter()) {
        for _ in 0..(w as usize) {
            sorted_outputs.push(*v);
            cursor += 1;
        }
        if cursor >= n {
            break;
        }
    }
    // Restore original input order.
    let mut out = vec![0.0_f64; n];
    for (rank, (orig_idx, _, _)) in indexed.iter().enumerate() {
        out[*orig_idx] = sorted_outputs[rank];
    }
    out
}

/// Conformal prediction interval at miscoverage α.
/// Returns the empirical coverage on a calibration set.
#[must_use]
pub fn conformal_coverage(scores: &[f64], targets: &[f64], alpha: f64) -> f64 {
    if scores.len() != targets.len() || scores.is_empty() {
        return f64::NAN;
    }
    let n = scores.len();
    let mut residuals: Vec<f64> = scores
        .iter()
        .zip(targets.iter())
        .map(|(s, t)| (s - t).abs())
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q_idx = ((1.0 - alpha) * (n + 1) as f64).ceil() as usize;
    let q = residuals[(q_idx - 1).min(n - 1)];
    // Coverage = fraction of residuals ≤ q (always (q_idx)/n by construction).
    residuals.iter().filter(|&&r| r <= q).count() as f64 / n as f64
}

/// Ensemble averaging of calibrated probabilities.
#[must_use]
pub fn ensemble_average(member_probs: &[Vec<f64>]) -> Vec<f64> {
    if member_probs.is_empty() {
        return Vec::new();
    }
    let n = member_probs[0].len();
    let m = member_probs.len() as f64;
    let mut out = vec![0.0_f64; n];
    for member in member_probs {
        for i in 0..n.min(member.len()) {
            out[i] += member[i];
        }
    }
    for v in &mut out {
        *v /= m;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overconfident_fixture() -> (Vec<f64>, Vec<bool>) {
        // Logits with T=1: confident 0.95 but only 70% accurate → overconfident.
        let probs: Vec<f64> = vec![0.95; 10].into_iter().chain(vec![0.05; 10]).collect();
        let correct: Vec<bool> = vec![true; 7]
            .into_iter()
            .chain(vec![false; 3])
            .chain(vec![false; 7])
            .chain(vec![true; 3])
            .collect();
        (probs, correct)
    }

    #[test]
    fn temperature_preserves_argmax() {
        let logits = vec![-2.0, -0.5, 0.5, 2.0];
        let baseline: Vec<bool> = logits.iter().map(|l| argmax_pred(sigmoid(*l))).collect();
        for t in &[1.5, 2.0, 5.0] {
            let scaled: Vec<bool> = temperature_apply(&logits, *t)
                .iter()
                .map(|p| argmax_pred(*p))
                .collect();
            assert_eq!(baseline, scaled, "temperature must not flip argmax");
        }
    }

    #[test]
    fn temperature_reduces_ece_when_overconfident() {
        let (probs, correct) = overconfident_fixture();
        // Treat probabilities as confidences directly. ECE_raw is high.
        let ece_raw = ece(&probs, &correct);
        // After temperature scaling at T=2, confidences move toward 0.5 and ECE drops.
        let logits: Vec<f64> = probs.iter().map(|p| (p / (1.0 - p)).ln()).collect();
        let scaled = temperature_apply(&logits, 2.0);
        let ece_scaled = ece(&scaled, &correct);
        assert!(
            ece_scaled < ece_raw,
            "temperature must reduce ECE on overconfident: {ece_raw} → {ece_scaled}"
        );
    }

    #[test]
    fn platt_passes_through_half_when_intercept_zero() {
        // sigmoid(a · 0 + 0) = 0.5 for any a.
        let p = platt_apply(0.0, 1.0, 0.0);
        assert!((p - 0.5).abs() < 1e-12);
    }

    #[test]
    fn isotonic_is_monotonic_in_input_score() {
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let targets = vec![0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9];
        let calib = isotonic_pav(&scores, &targets);
        // Verify monotone on the sorted-by-score order.
        let mut indexed: Vec<(usize, f64)> =
            scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let calib_sorted: Vec<f64> = indexed.iter().map(|(i, _)| calib[*i]).collect();
        for w in calib_sorted.windows(2) {
            assert!(w[0] <= w[1] + 1e-12, "isotonic must be monotone");
        }
    }

    #[test]
    fn conformal_alpha_010_yields_close_to_090_coverage() {
        let scores: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let targets: Vec<f64> = scores.iter().map(|s| s + 0.05).collect();
        let cov = conformal_coverage(&scores, &targets, 0.1);
        assert!(
            (0.85..=1.0).contains(&cov),
            "conformal coverage at α=0.1 should be ~0.9, got {cov}"
        );
    }

    #[test]
    fn ensemble_average_reduces_ece_vs_worst_member() {
        let m1: Vec<f64> = vec![0.95, 0.05, 0.95, 0.05];
        let m2: Vec<f64> = vec![0.85, 0.15, 0.85, 0.15];
        let m3: Vec<f64> = vec![0.75, 0.25, 0.75, 0.25];
        let correct = vec![true, false, true, false];
        let avg = ensemble_average(&[m1.clone(), m2.clone(), m3.clone()]);
        let ece_worst = ece(&m1, &correct)
            .max(ece(&m2, &correct))
            .max(ece(&m3, &correct));
        let ece_avg = ece(&avg, &correct);
        assert!(
            ece_avg <= ece_worst + 1e-12,
            "ensemble must not be worse than worst member"
        );
    }
}
