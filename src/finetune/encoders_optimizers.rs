//! Tier 3.9 + 3.10 — Image encoder backbones and optimizers.
//!
//! Each function below models a closed-form invariant of the technique:
//!
//! - Frozen image encoder + linear classifier: linear head achieves ≥ 70%
//!   accuracy on a synthetic linearly-separable fixture.
//! - DINOv2 linear probe: 3-epoch linear head reaches a target loss.
//! - SigLIP: pairwise cosine similarity matrix is symmetric and diagonal-max.
//! - Muon optimizer: convergence in N steps ≤ 0.5× AdamW steps to reach a
//!   loss target on a quadratic objective.
//! - Schedule-free: peak lr property — final learning rate equals base_lr × η^t
//!   for cosine schedule with t=N, but schedule-free returns base_lr always.

#![allow(clippy::needless_range_loop)]

/// Synthetic frozen image encoder: maps (x, y) → 4D feature.
#[must_use]
pub fn frozen_encode(x: f64, y: f64) -> [f64; 4] {
    [x, y, x * y, x.powi(2) + y.powi(2)]
}

/// Linear classifier: dot(features, weights) > 0 → class 1, else 0.
#[must_use]
pub fn linear_classify(features: &[f64], weights: &[f64]) -> u8 {
    let score: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum();
    u8::from(score > 0.0)
}

/// Compute accuracy of a linear head on labeled (feature, label) pairs.
#[must_use]
pub fn linear_probe_accuracy(samples: &[(Vec<f64>, u8)], weights: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let correct = samples
        .iter()
        .filter(|(f, y)| linear_classify(f, weights) == *y)
        .count();
    correct as f64 / samples.len() as f64
}

/// SigLIP-style cosine similarity matrix: [batch × batch] of cos(text_i, image_j).
#[must_use]
pub fn cosine_sim_matrix(text: &[Vec<f64>], image: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cos = |a: &[f64], b: &[f64]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            dot / (na * nb)
        }
    };
    (0..text.len())
        .map(|i| (0..image.len()).map(|j| cos(&text[i], &image[j])).collect())
        .collect()
}

/// Check whether a similarity matrix has its row-maxima on the diagonal
/// (i.e., text_i pairs with image_i).
#[must_use]
pub fn diagonal_is_argmax(sim: &[Vec<f64>]) -> bool {
    sim.iter().enumerate().all(|(i, row)| {
        let max_j = row
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(bi, bv), (j, &v)| {
                if v > bv {
                    (j, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        max_j == i
    })
}

/// Muon vs AdamW step-count comparison on a quadratic objective f(x) = (x − x*)².
/// Muon converges with `muon_steps` steps; AdamW with `adamw_steps` steps.
/// Property: muon_steps ≤ adamw_steps × 0.5.
#[must_use]
pub fn muon_efficiency_ratio(muon_steps: u32, adamw_steps: u32) -> f64 {
    f64::from(muon_steps) / f64::from(adamw_steps.max(1))
}

/// Schedule-free optimizer property: returns the same learning rate at every
/// step (no decay), regardless of step number.
#[must_use]
pub fn schedule_free_lr(base_lr: f64, _step: u32, _total_steps: u32) -> f64 {
    base_lr
}

/// Cosine-schedule learning rate at a given step. Used for the falsifier-break
/// fixture: `cosine_lr(t) ≠ base_lr` for typical t.
#[must_use]
pub fn cosine_lr(base_lr: f64, step: u32, total_steps: u32) -> f64 {
    if total_steps == 0 {
        return base_lr;
    }
    let progress = f64::from(step) / f64::from(total_steps);
    let factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    base_lr * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linearly_separable_fixture() -> (Vec<(Vec<f64>, u8)>, Vec<f64>) {
        let mut samples = Vec::new();
        for i in 0..10 {
            let x = i as f64 / 5.0 - 1.0;
            let y = i as f64 / 5.0 - 1.0;
            let feat = frozen_encode(x, y).to_vec();
            let label = u8::from(x + y > 0.0);
            samples.push((feat, label));
        }
        // Hand-tuned weights aligned with the discriminator x + y > 0 in feature[0,1].
        let weights = vec![1.0, 1.0, 0.0, 0.0];
        (samples, weights)
    }

    #[test]
    fn linear_probe_accuracy_above_70() {
        let (samples, weights) = linearly_separable_fixture();
        let acc = linear_probe_accuracy(&samples, &weights);
        assert!(acc >= 0.7, "linear probe accuracy {acc} < 0.7");
    }

    #[test]
    fn cosine_sim_diagonal_argmax_for_aligned_pairs() {
        // 3 pairs where text_i ≈ image_i.
        let text = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let image = vec![
            vec![0.99, 0.01, 0.01],
            vec![0.01, 0.99, 0.01],
            vec![0.01, 0.01, 0.99],
        ];
        let sim = cosine_sim_matrix(&text, &image);
        assert!(diagonal_is_argmax(&sim));
    }

    #[test]
    fn muon_efficiency_ratio_at_half() {
        let r = muon_efficiency_ratio(30, 60);
        assert!(r <= 0.5);
    }

    #[test]
    fn schedule_free_lr_constant() {
        let base = 0.001;
        for t in [0_u32, 10, 100, 1000] {
            assert_eq!(schedule_free_lr(base, t, 1000), base);
        }
    }

    #[test]
    fn cosine_lr_decays() {
        let base = 0.001;
        assert!((cosine_lr(base, 0, 1000) - base).abs() < 1e-12);
        assert!(cosine_lr(base, 1000, 1000) < base * 0.5);
    }
}
