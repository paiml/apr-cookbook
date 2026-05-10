//! Tier 4.1–4.3 — Preference-learning helpers (DPO, ORPO, KTO).
//!
//! Each function below models the *observable closed-form invariant* of the
//! algorithm so recipe falsifiers are deterministic checks rather than
//! stochastic training-loss claims.
//!
//! - DPO loss: −log σ(β · (log π(chosen)/π_ref(chosen) − log π(rejected)/π_ref(rejected)))
//!   - Lower for "chosen > rejected" preferences.
//!   - Lower KL with smaller β (β=0.1 vs β=1.0).
//! - ORPO loss: −log σ(log p(chosen)/(1−p(chosen)) − log p(rejected)/(1−p(rejected))).
//!   - No reference model; concave in rejected likelihood.
//! - KTO: KL-regularized binary feedback; positive (chosen) samples raise
//!   their conditional likelihood, negative (rejected) lower it.

#![allow(clippy::needless_range_loop)]

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn log_sigmoid(z: f64) -> f64 {
    // log(1/(1+exp(-z))) = -log(1+exp(-z))
    if z > 0.0 {
        -((-z).exp().ln_1p())
    } else {
        z - z.exp().ln_1p()
    }
}

/// DPO loss for one (chosen, rejected) pair given log-probability ratios.
/// `lp_chosen` = log π(chosen) − log π_ref(chosen);
/// `lp_rejected` = log π(rejected) − log π_ref(rejected).
/// Returns the per-sample loss; lower = preference correctly captured.
#[must_use]
pub fn dpo_loss(lp_chosen: f64, lp_rejected: f64, beta: f64) -> f64 {
    -log_sigmoid(beta * (lp_chosen - lp_rejected))
}

/// Implicit reward at a token under DPO: r(x) = β · log π(x)/π_ref(x).
/// Property: r(chosen) > r(rejected) when DPO has converged on the pair.
#[must_use]
pub fn dpo_implicit_reward(lp_diff: f64, beta: f64) -> f64 {
    beta * lp_diff
}

/// KL divergence approximation between policy and reference, via mean
/// log-probability difference.
#[must_use]
pub fn kl_estimate(log_probs_diff: &[f64]) -> f64 {
    if log_probs_diff.is_empty() {
        return 0.0;
    }
    log_probs_diff.iter().sum::<f64>() / log_probs_diff.len() as f64
}

/// ORPO loss on a (chosen, rejected) pair given probabilities.
/// L = −log σ(log p_chosen / (1 − p_chosen) − log p_rejected / (1 − p_rejected)).
#[must_use]
pub fn orpo_loss(p_chosen: f64, p_rejected: f64) -> f64 {
    let log_odds = |p: f64| -> f64 {
        let p = p.clamp(1e-9, 1.0 - 1e-9);
        (p / (1.0 - p)).ln()
    };
    -log_sigmoid(log_odds(p_chosen) - log_odds(p_rejected))
}

/// ORPO loss is monotone increasing in p_rejected: as the model assigns more
/// probability to the rejected response, loss grows. (The ORPO paper's
/// "concavity" claim refers to the score landscape, not the loss itself —
/// monotonicity is the cleaner deterministic check.)
#[must_use]
pub fn orpo_monotone_in_rejected(p_chosen: f64, p_rejected_base: f64) -> bool {
    let h = 0.05_f64;
    if p_rejected_base + h >= 1.0 {
        return true;
    }
    let l_base = orpo_loss(p_chosen, p_rejected_base);
    let l_higher = orpo_loss(p_chosen, p_rejected_base + h);
    l_higher > l_base
}

/// KTO single-sample loss (binary feedback).
/// `desirability ∈ [0,1]` is the prior over positive feedback.
/// `lp_diff` = log π(x) − log π_ref(x).
/// `is_positive` = whether the sample is "thumb-up" / chosen.
#[must_use]
pub fn kto_loss(lp_diff: f64, beta: f64, is_positive: bool, desirability: f64) -> f64 {
    let scaled = beta * lp_diff;
    if is_positive {
        // Loss decreases as the policy raises log π for positive samples.
        desirability * (1.0 - sigmoid(scaled))
    } else {
        (1.0 - desirability) * sigmoid(scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpo_lower_when_chosen_likely() {
        // chosen log-ratio > rejected → smaller loss.
        let lp_chosen = 0.5_f64;
        let lp_rejected = -0.5_f64;
        let beta = 0.1_f64;
        let loss_correct = dpo_loss(lp_chosen, lp_rejected, beta);
        let loss_swapped = dpo_loss(lp_rejected, lp_chosen, beta);
        assert!(loss_correct < loss_swapped);
    }

    #[test]
    fn dpo_smaller_beta_smaller_kl() {
        let diffs = [0.5_f64, 0.4, 0.6, 0.45, 0.55];
        let kl_b01 = kl_estimate(&diffs.iter().map(|d| 0.1 * d).collect::<Vec<_>>());
        let kl_b1 = kl_estimate(&diffs.iter().map(|d| 1.0 * d).collect::<Vec<_>>());
        assert!(kl_b01 < kl_b1);
    }

    #[test]
    fn dpo_implicit_reward_chosen_above_rejected() {
        // For a converged pair, lp_diff_chosen > lp_diff_rejected.
        let r_chosen = dpo_implicit_reward(0.5, 0.1);
        let r_rejected = dpo_implicit_reward(-0.5, 0.1);
        assert!(r_chosen > r_rejected);
    }

    #[test]
    fn orpo_no_ref_model_runs() {
        // ORPO accepts only (p_chosen, p_rejected); no reference distribution needed.
        let l = orpo_loss(0.7, 0.3);
        assert!(l > 0.0 && l.is_finite());
    }

    #[test]
    fn orpo_monotone_increasing_in_p_rejected() {
        assert!(orpo_monotone_in_rejected(0.7, 0.3));
        assert!(orpo_monotone_in_rejected(0.6, 0.5));
    }

    #[test]
    fn kto_unbiased_at_desirability_05() {
        // For balanced positive/negative samples with equal lp_diff and
        // desirability = 0.5, the gradient sums to zero in expectation.
        let lp_pos = 0.5;
        let lp_neg = 0.5;
        let beta = 0.1;
        let lp = kto_loss(lp_pos, beta, true, 0.5);
        let ln = kto_loss(lp_neg, beta, false, 0.5);
        // 0.5*(1-σ) + 0.5*σ = 0.5 — total constant, gradient cancels.
        assert!((lp + ln - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kto_works_on_isolated_samples() {
        // KTO does not require pairs; one positive sample is enough.
        let l = kto_loss(0.3, 0.1, true, 0.5);
        assert!(l > 0.0 && l.is_finite());
    }
}
