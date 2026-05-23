//! Tier 3.5 Multimodal + multitask + k-fold — shared helper.
//!
//! Each function below models a closed-form invariant of the technique:
//!
//! - text+image fusion: combined feature vector = concat(text_feat, image_feat)
//!   with dim = d_text + d_image.
//! - text+tabular fusion: combined predictor uses both features so its
//!   representational capacity ≥ either alone.
//! - multitask: per-task loss must decrease independently; total loss is
//!   weighted sum (no catastrophic forgetting on any task).
//! - zero-shot: argmax of per-class log-likelihood ranking matches expected
//!   class on a deterministic fixture.
//! - k-fold CV: K folds produce a partition (disjoint, complete) of the data.

#![allow(clippy::needless_range_loop)]

/// Text-image fusion: concatenate features along dim axis.
#[must_use]
pub fn fuse_concat(text_feat: &[f64], image_feat: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(text_feat.len() + image_feat.len());
    out.extend_from_slice(text_feat);
    out.extend_from_slice(image_feat);
    out
}

/// Text-tabular fusion via gated combination: y = σ(g)·text + (1−σ(g))·tabular.
#[must_use]
pub fn fuse_gated(text: &[f64], tabular: &[f64], gate: f64) -> Vec<f64> {
    if text.len() != tabular.len() {
        return Vec::new();
    }
    let g = sigmoid(gate);
    text.iter()
        .zip(tabular.iter())
        .map(|(t, x)| g * t + (1.0 - g) * x)
        .collect()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Multitask SFT step: each task t has loss L_t. After a step, per-task
/// loss decreases monotonically (no catastrophic forgetting).
#[derive(Debug, Clone, PartialEq)]
pub struct MultitaskStep {
    pub task_losses_before: Vec<f64>,
    pub task_losses_after: Vec<f64>,
    pub weights: Vec<f64>,
}

impl MultitaskStep {
    /// True if every task's loss decreased.
    #[must_use]
    pub fn all_decreased(&self) -> bool {
        self.task_losses_after
            .iter()
            .zip(self.task_losses_before.iter())
            .all(|(a, b)| a < b)
    }

    /// Weighted total loss before/after.
    #[must_use]
    pub fn total_before(&self) -> f64 {
        self.weighted_sum(&self.task_losses_before)
    }

    #[must_use]
    pub fn total_after(&self) -> f64 {
        self.weighted_sum(&self.task_losses_after)
    }

    fn weighted_sum(&self, losses: &[f64]) -> f64 {
        losses
            .iter()
            .zip(self.weights.iter())
            .map(|(l, w)| l * w)
            .sum()
    }
}

/// Build a single multitask "training step": each task makes 10% progress.
#[must_use]
pub fn synthetic_multitask_step(initial_losses: &[f64], weights: &[f64]) -> MultitaskStep {
    let after: Vec<f64> = initial_losses.iter().map(|l| l * 0.9).collect();
    MultitaskStep {
        task_losses_before: initial_losses.to_vec(),
        task_losses_after: after,
        weights: weights.to_vec(),
    }
}

/// Zero-shot classifier: pick argmax of per-class log-likelihoods on a query.
#[must_use]
pub fn zero_shot_predict(class_log_probs: &[f64]) -> usize {
    class_log_probs
        .iter()
        .enumerate()
        .fold((0_usize, f64::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i, v)
            } else {
                (best_i, best_v)
            }
        })
        .0
}

/// K-fold CV: returns a list of (train_indices, val_indices) pairs.
/// Stratified-disjoint by index: fold k contains indices i where i % K == k.
#[must_use]
pub fn kfold_split(n: usize, k: u32) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut out = Vec::with_capacity(k as usize);
    for fold in 0..k {
        let mut train = Vec::new();
        let mut val = Vec::new();
        for i in 0..n {
            if (i as u32) % k == fold {
                val.push(i);
            } else {
                train.push(i);
            }
        }
        out.push((train, val));
    }
    out
}

/// Verify that K validation sets form a partition: union covers 0..n,
/// pairwise intersections are empty.
#[must_use]
pub fn kfold_is_partition(n: usize, k: u32) -> bool {
    let folds = kfold_split(n, k);
    let mut union: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (_, val) in &folds {
        let val_set: std::collections::HashSet<usize> = val.iter().copied().collect();
        if !union.is_disjoint(&val_set) {
            return false;
        }
        union.extend(val_set);
    }
    union.len() == n && (0..n).all(|i| union.contains(&i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_concat_dim_equals_sum() {
        let text = vec![1.0, 2.0, 3.0];
        let image = vec![4.0, 5.0];
        let fused = fuse_concat(&text, &image);
        assert_eq!(fused.len(), text.len() + image.len());
    }

    #[test]
    fn gated_fusion_at_gate_zero_is_tabular() {
        // sigmoid(-large) ≈ 0 → fused ≈ tabular.
        let text = vec![1.0, 2.0];
        let tab = vec![10.0, 20.0];
        let fused = fuse_gated(&text, &tab, -10.0);
        assert!((fused[0] - tab[0]).abs() < 0.01);
        assert!((fused[1] - tab[1]).abs() < 0.01);
    }

    #[test]
    fn multitask_no_catastrophic_forgetting() {
        let step = synthetic_multitask_step(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        assert!(step.all_decreased());
        assert!(step.total_after() < step.total_before());
    }

    #[test]
    fn zero_shot_picks_max_log_prob_class() {
        let log_probs = vec![-2.5, -1.0, -3.0, -0.5];
        // Class 3 has highest log-prob.
        assert_eq!(zero_shot_predict(&log_probs), 3);
    }

    #[test]
    fn kfold_5_disjoint_covers_data() {
        assert!(kfold_is_partition(50, 5));
        let folds = kfold_split(50, 5);
        assert_eq!(folds.len(), 5);
        for (_, val) in &folds {
            assert_eq!(val.len(), 10);
        }
    }

    #[test]
    fn kfold_10_disjoint_covers_data() {
        assert!(kfold_is_partition(100, 10));
    }
}
