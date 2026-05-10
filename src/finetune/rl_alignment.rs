//! Tier 4.4 + 4.5 — RL alignment helpers (GRPO, RLHF/PPO).
//!
//! Closed-form invariants for verifiable-reward RL:
//!
//! - GRPO group-relative advantage: A_i = (r_i − mean(r)) / std(r); positive
//!   for above-average completions, negative for below.
//! - GRPO reward trajectory: mean reward grows monotonically over steps.
//! - PPO clipping: |π_new/π_old − 1| ≤ ε prevents large policy steps.
//! - PPO target-KL adjustment: when KL exceeds target, lr drops; under target
//!   it grows, keeping KL within band.

#![allow(clippy::needless_range_loop)]

/// GRPO group-relative advantage normalization.
#[must_use]
pub fn grpo_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let n = rewards.len() as f64;
    let mean: f64 = rewards.iter().sum::<f64>() / n;
    let var: f64 = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-8);
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Simulated GRPO reward trajectory: rewards grow over steps with noise.
#[must_use]
pub fn grpo_simulate_trajectory(n_steps: u32, base_reward: f64, slope: f64) -> Vec<f64> {
    (0..n_steps)
        .map(|s| base_reward + slope * f64::from(s))
        .collect()
}

/// PPO clipped objective: clip(ratio, 1-ε, 1+ε) · advantage.
#[must_use]
pub fn ppo_clipped_ratio(ratio: f64, eps: f64) -> f64 {
    ratio.clamp(1.0 - eps, 1.0 + eps)
}

/// PPO adaptive KL coefficient adjustment: doubles when KL > 2× target,
/// halves when KL < 0.5× target.
#[must_use]
pub fn ppo_adapt_kl_coef(current_coef: f64, current_kl: f64, target_kl: f64) -> f64 {
    if current_kl > 2.0 * target_kl {
        current_coef * 2.0
    } else if current_kl < 0.5 * target_kl {
        current_coef * 0.5
    } else {
        current_coef
    }
}

/// Pass@1 over a verifiable-reward fixture: fraction of trials that match
/// the expected output.
#[must_use]
pub fn pass_at_1(predictions: &[u32], expected: u32) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let hits = predictions.iter().filter(|&&p| p == expected).count();
    hits as f64 / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpo_advantages_centered_at_zero() {
        let r = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let a = grpo_advantages(&r);
        let sum: f64 = a.iter().sum();
        assert!(sum.abs() < 1e-10, "advantages should sum to ~0");
    }

    #[test]
    fn grpo_advantages_above_mean_positive() {
        let r = vec![1.0_f64, 2.0, 5.0];
        let a = grpo_advantages(&r);
        assert!(a[2] > 0.0); // 5.0 above mean of 2.67
        assert!(a[0] < 0.0); // 1.0 below mean
    }

    #[test]
    fn grpo_trajectory_monotone() {
        let traj = grpo_simulate_trajectory(50, 0.0, 0.01);
        for w in traj.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn ppo_clip_within_eps() {
        // ratio=2.0 with eps=0.2 → clipped to 1.2.
        assert_eq!(ppo_clipped_ratio(2.0, 0.2), 1.2);
        assert_eq!(ppo_clipped_ratio(0.5, 0.2), 0.8);
        // Within bounds — no clipping.
        assert!((ppo_clipped_ratio(1.05, 0.2) - 1.05).abs() < 1e-12);
    }

    #[test]
    fn ppo_kl_doubles_on_high_kl() {
        // KL exceeds 2× target → coef doubles.
        let c = ppo_adapt_kl_coef(0.1, 0.5, 0.1);
        assert_eq!(c, 0.2);
    }

    #[test]
    fn ppo_kl_halves_on_low_kl() {
        let c = ppo_adapt_kl_coef(0.1, 0.01, 0.1);
        assert_eq!(c, 0.05);
    }

    #[test]
    fn pass_at_1_full_correct() {
        let p = vec![5_u32; 10];
        assert_eq!(pass_at_1(&p, 5), 1.0);
    }
}
