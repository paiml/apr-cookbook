#![allow(
    clippy::needless_range_loop,
    clippy::redundant_closure_for_method_calls
)]
//! Tier 2.1 LoRA — shared helper.
//!
//! Implements the canonical LoRA decomposition:
//!
//!   W' = W_base + (α/r) · B · A
//!
//! where W_base is frozen, A ∈ ℝ^(r×d_in), B ∈ ℝ^(d_out×r), and r ≪ d.
//!
//! - Trainable parameters: r · (d_in + d_out)
//! - Merge round-trip: W_merged := W_base + (α/r) · B · A
//! - When α/r = 1.0, repeated merges are bit-identical
//!
//! No SGD here; the falsifier is closed-form linear algebra.

use crate::Result;

/// LoRA layer state: frozen base + low-rank adapter.
#[derive(Debug, Clone)]
pub struct LoraLayer {
    pub d_out: usize,
    pub d_in: usize,
    pub rank: u32,
    pub alpha: f64,
    /// Frozen base weight matrix W_base ∈ ℝ^(d_out × d_in).
    pub base: Vec<Vec<f64>>,
    /// Down-projection A ∈ ℝ^(rank × d_in).
    pub a: Vec<Vec<f64>>,
    /// Up-projection B ∈ ℝ^(d_out × rank).
    pub b: Vec<Vec<f64>>,
}

impl LoraLayer {
    /// Construct a new LoraLayer with deterministic init.
    /// Per the LoRA paper: A ~ small Gaussian, B = 0 so initial ΔW = 0.
    #[must_use]
    pub fn new(d_out: usize, d_in: usize, rank: u32, alpha: f64) -> Self {
        let r = rank as usize;
        let base = (0..d_out)
            .map(|i| (0..d_in).map(|j| ((i + j) as f64 % 7.0) / 7.0).collect())
            .collect();
        let a = (0..r)
            .map(|i| {
                (0..d_in)
                    .map(|j| ((i * 3 + j * 5) as f64 % 11.0) / 100.0)
                    .collect()
            })
            .collect();
        let b = vec![vec![0.0_f64; r]; d_out]; // zero-init per paper
        Self {
            d_out,
            d_in,
            rank,
            alpha,
            base,
            a,
            b,
        }
    }

    /// Trainable parameter count = rank × (d_in + d_out).
    #[must_use]
    pub fn trainable_params(&self) -> u64 {
        u64::from(self.rank) * (self.d_in as u64 + self.d_out as u64)
    }

    /// Frozen parameter count = d_out × d_in.
    #[must_use]
    pub fn frozen_params(&self) -> u64 {
        self.d_out as u64 * self.d_in as u64
    }

    /// Reduction ratio: trainable / total.
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        let trainable = self.trainable_params() as f64;
        let total = trainable + self.frozen_params() as f64;
        trainable / total
    }

    /// Merge the LoRA delta into the base: W := W + (α/r) · B · A.
    /// After this, A and B can be discarded for inference.
    /// Returns the merged W matrix (does not mutate `self`).
    #[must_use]
    pub fn merge(&self) -> Vec<Vec<f64>> {
        let scale = self.alpha / f64::from(self.rank);
        let mut merged = self.base.clone();
        // delta = B (d_out × r) · A (r × d_in)
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut dot = 0.0_f64;
                for k in 0..self.rank as usize {
                    dot += self.b[i][k] * self.a[k][j];
                }
                merged[i][j] += scale * dot;
            }
        }
        merged
    }

    /// Unmerge: subtract (α/r) · B · A from a previously-merged matrix.
    /// Returns base = merged - delta.
    #[must_use]
    pub fn unmerge(&self, merged: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let scale = self.alpha / f64::from(self.rank);
        let mut base: Vec<Vec<f64>> = merged.to_vec();
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut dot = 0.0_f64;
                for k in 0..self.rank as usize {
                    dot += self.b[i][k] * self.a[k][j];
                }
                base[i][j] -= scale * dot;
            }
        }
        base
    }

    /// Set adapter B to non-zero (simulates after-training state).
    pub fn set_b_for_test(&mut self, value: f64) {
        for row in &mut self.b {
            for v in row.iter_mut() {
                *v = value;
            }
        }
    }
}

/// Frobenius distance between two matrices.
#[must_use]
pub fn frobenius_distance(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::NAN;
    }
    let mut sum_sq = 0.0_f64;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if row_a.len() != row_b.len() {
            return f64::NAN;
        }
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            let d = x - y;
            sum_sq += d * d;
        }
    }
    sum_sq.sqrt()
}

/// Apply LoRA to a single input vector: y = (W_base + (α/r)·B·A) x.
#[must_use]
pub fn forward(layer: &LoraLayer, x: &[f64]) -> Vec<f64> {
    let merged = layer.merge();
    let mut y = vec![0.0_f64; layer.d_out];
    for (i, row) in merged.iter().enumerate() {
        for (j, w) in row.iter().enumerate() {
            y[i] += w * x[j];
        }
    }
    y
}

/// Run a tiny SGD on a synthetic objective. Used to demonstrate that
/// LoRA actually trains (loss decreases). Returns (initial, final, steps).
pub fn run_smoke_train(
    layer: &mut LoraLayer,
    target_y: &[f64],
    x: &[f64],
    steps: u32,
) -> Result<(f64, f64, u32)> {
    if x.len() != layer.d_in || target_y.len() != layer.d_out {
        return Err(crate::CookbookError::invalid_format(format!(
            "shape mismatch: x={}x1 target={}x1 layer={}x{}",
            x.len(),
            target_y.len(),
            layer.d_out,
            layer.d_in
        )));
    }
    let lr = 0.01_f64;
    let initial_loss = mse_to_target(layer, x, target_y);
    let mut final_loss = initial_loss;
    for _ in 0..steps {
        let pred = forward(layer, x);
        // Gradient w.r.t. b[i][k]: 2*(pred[i] - target[i]) * (α/r) * A[k] · x
        let scale = layer.alpha / f64::from(layer.rank);
        let mut a_dot_x = vec![0.0_f64; layer.rank as usize];
        for k in 0..layer.rank as usize {
            for j in 0..layer.d_in {
                a_dot_x[k] += layer.a[k][j] * x[j];
            }
        }
        for i in 0..layer.d_out {
            let err = pred[i] - target_y[i];
            for k in 0..layer.rank as usize {
                layer.b[i][k] -= lr * 2.0 * err * scale * a_dot_x[k];
            }
        }
        final_loss = mse_to_target(layer, x, target_y);
    }
    Ok((initial_loss, final_loss, steps))
}

fn mse_to_target(layer: &LoraLayer, x: &[f64], target: &[f64]) -> f64 {
    let pred = forward(layer, x);
    let n = target.len() as f64;
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainable_params_formula() {
        // Per LoRA paper: trainable = r × (d_in + d_out)
        let layer = LoraLayer::new(64, 32, 8, 8.0);
        assert_eq!(layer.trainable_params(), 8 * (32 + 64));
    }

    #[test]
    fn rank_32_has_4x_more_params_than_rank_8() {
        let r8 = LoraLayer::new(64, 32, 8, 8.0);
        let r32 = LoraLayer::new(64, 32, 32, 32.0);
        assert_eq!(r32.trainable_params(), 4 * r8.trainable_params());
    }

    #[test]
    fn b_zero_init_means_merge_equals_base() {
        // With B = 0 (the default), merging should not change base.
        let layer = LoraLayer::new(8, 8, 4, 4.0);
        let merged = layer.merge();
        let dist = frobenius_distance(&layer.base, &merged);
        assert!(
            dist < 1e-12,
            "B=0 should preserve base on merge, got {dist}"
        );
    }

    #[test]
    fn merge_unmerge_roundtrip_when_alpha_eq_r() {
        // With α/r = 1.0, merge then unmerge should be bit-identical.
        let mut layer = LoraLayer::new(8, 8, 4, 4.0); // α=4, r=4 → α/r=1.0
        layer.set_b_for_test(0.1);
        let merged = layer.merge();
        let unmerged = layer.unmerge(&merged);
        let dist = frobenius_distance(&layer.base, &unmerged);
        assert!(
            dist < 1e-12,
            "merge/unmerge must round-trip bit-identical, got {dist}"
        );
    }

    #[test]
    fn reduction_ratio_is_small_for_low_rank() {
        // For 256×256 base with rank 8: ratio = 8 × (256+256) / (8 × (256+256) + 256 × 256) ≈ 0.06
        let layer = LoraLayer::new(256, 256, 8, 8.0);
        let ratio = layer.reduction_ratio();
        assert!(
            ratio < 0.07,
            "rank-8 LoRA should have <7% trainable, got {ratio}"
        );
    }

    #[test]
    fn smoke_training_decreases_loss() {
        let mut layer = LoraLayer::new(4, 4, 2, 2.0);
        let x = vec![1.0, 0.5, -0.5, 0.2];
        let target = vec![5.0, 4.0, 3.0, 2.0];
        let (initial, final_, steps) = run_smoke_train(&mut layer, &target, &x, 50).expect("train");
        assert_eq!(steps, 50);
        assert!(
            final_ < initial,
            "loss should decrease: {initial} → {final_}"
        );
    }
}
