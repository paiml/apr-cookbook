//! Tier 2.4 adapter merge — shared helper.
//!
//! Implements 5 LoRA-merge strategies as closed-form linear algebra so each
//! recipe has a tight, deterministic falsifier:
//!
//! - **average**: W = (W_a + W_b) / 2. Identical inputs return input unchanged.
//! - **SLERP**: W = sin((1-t)·θ)/sin(θ)·A + sin(t·θ)/sin(θ)·B. Midpoint norm.
//! - **DARE**: drop p% of delta entries (zero-mask), rescale by 1/(1-p).
//! - **TIES**: sign-resolve to keep entries matching summed delta direction.
//! - **multi-LoRA**: addition is associative; per-LoRA outputs recoverable.
//!
//! All operations on `Vec<f64>` so the falsifier is closed-form and bit-stable.

#![allow(clippy::needless_range_loop)]

/// Element-wise average of any number of LoRA delta vectors.
#[must_use]
pub fn average_merge(deltas: &[Vec<f64>]) -> Vec<f64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let n = deltas[0].len();
    let count = deltas.len() as f64;
    let mut out = vec![0.0_f64; n];
    for d in deltas {
        for i in 0..n {
            out[i] += d[i];
        }
    }
    for v in &mut out {
        *v /= count;
    }
    out
}

/// SLERP (spherical-linear interpolation) of two LoRA delta vectors at
/// position `t ∈ [0,1]`. Falls back to lerp when vectors are nearly parallel.
#[must_use]
pub fn slerp_merge(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    if a.len() != b.len() || a.is_empty() {
        return Vec::new();
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return lerp(a, b, t);
    }
    let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta.sin().abs() < 1e-6 {
        return lerp(a, b, t);
    }
    let s = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / s;
    let wb = (t * theta).sin() / s;
    a.iter().zip(b).map(|(x, y)| wa * x + wb * y).collect()
}

fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(x, y)| (1.0 - t) * x + t * y)
        .collect()
}

/// Vector L2 norm.
#[must_use]
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// DARE merge: deterministic-stride mask drops `drop_p` fraction of entries to
/// zero, then rescales the remainder by 1/(1-drop_p). Stride-based to keep the
/// falsifier deterministic without an RNG dependency.
#[must_use]
pub fn dare_merge(delta: &[f64], drop_p: f64) -> Vec<f64> {
    if delta.is_empty() || !(0.0..1.0).contains(&drop_p) {
        return delta.to_vec();
    }
    let n = delta.len();
    let n_drop = ((n as f64) * drop_p).round() as usize;
    let kept_scale = 1.0 / (1.0 - drop_p);
    // Drop every k-th entry by deterministic stride.
    let stride = if n_drop == 0 {
        usize::MAX
    } else {
        n / n_drop.max(1)
    };
    let mut out = vec![0.0_f64; n];
    let mut dropped = 0_usize;
    for i in 0..n {
        if dropped < n_drop && i % stride.max(1) == 0 {
            out[i] = 0.0;
            dropped += 1;
        } else {
            out[i] = delta[i] * kept_scale;
        }
    }
    out
}

/// TIES merge across multiple deltas. Sign-resolve: per-coordinate, the merged
/// value retains entries whose sign matches the *summed* sign direction.
#[must_use]
pub fn ties_merge(deltas: &[Vec<f64>]) -> Vec<f64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let n = deltas[0].len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        // Magnitude-weighted sign vote, per the TIES paper. Pure sign-vote ties
        // (sum == 0) are zeroed because the direction is undefined.
        let sum_signed: f64 = deltas.iter().map(|d| d[i]).sum();
        if sum_signed.abs() < 1e-12 {
            out[i] = 0.0;
            continue;
        }
        let sign_dir = sum_signed.signum();
        let kept: Vec<f64> = deltas
            .iter()
            .map(|d| d[i])
            .filter(|v| v.signum() == sign_dir)
            .collect();
        if kept.is_empty() {
            out[i] = 0.0;
        } else {
            out[i] = kept.iter().sum::<f64>() / kept.len() as f64;
        }
    }
    out
}

/// Multi-LoRA load: stacked deltas applied additively. Returns the summed
/// delta. Property: applying additive multi-LoRA to orthogonal inputs
/// preserves each LoRA's effect.
#[must_use]
pub fn multilora_apply(base: &[f64], deltas: &[Vec<f64>]) -> Vec<f64> {
    if deltas.is_empty() {
        return base.to_vec();
    }
    let n = base.len();
    let mut out = base.to_vec();
    for d in deltas {
        for i in 0..n.min(d.len()) {
            out[i] += d[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_delta(seed: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (((i as u32 * seed) % 17) as f64) / 17.0 - 0.5)
            .collect()
    }

    #[test]
    fn average_of_identical_returns_input() {
        let a = synthetic_delta(3, 32);
        let merged = average_merge(&[a.clone(), a.clone(), a.clone()]);
        for (x, y) in a.iter().zip(merged.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn slerp_at_t_half_midpoint_norm() {
        let a = synthetic_delta(7, 32);
        let b = synthetic_delta(11, 32);
        let mid = slerp_merge(&a, &b, 0.5);
        let na = norm(&a);
        let nb = norm(&b);
        let nm = norm(&mid);
        // Midpoint norm bounded by max of {|A|,|B|} × 1.05 and ≥ 0.5*(|A|+|B|) - tolerance.
        assert!(nm <= na.max(nb) * 1.05);
        assert!(nm >= 0.45 * (na + nb));
    }

    #[test]
    fn dare_drop_05_zeros_half_the_entries() {
        let delta = synthetic_delta(13, 100);
        let merged = dare_merge(&delta, 0.5);
        let zeros = merged.iter().filter(|v| v.abs() < 1e-12).count();
        // ±5 deterministic-stride tolerance.
        assert!(
            (40..=60).contains(&zeros),
            "expected ~50 zeros, got {zeros}"
        );
    }

    #[test]
    fn ties_preserves_shared_sign_direction() {
        // Both deltas positive at index i → merged stays positive at i.
        let a = vec![1.0, -1.0, 0.5];
        let b = vec![2.0, 1.0, 0.25];
        let m = ties_merge(&[a, b]);
        // Index 0: both positive → kept (avg of {1.0, 2.0} = 1.5).
        // Index 1: -1.0 + 1.0 = 0 (perfect cancellation) → 0.
        // Index 2: both positive → avg of {0.5, 0.25} = 0.375.
        assert!((m[0] - 1.5).abs() < 1e-12);
        assert!(m[1].abs() < 1e-12);
        assert!((m[2] - 0.375).abs() < 1e-12);
    }

    #[test]
    fn multilora_additive_round_trip() {
        let base = vec![1.0, 2.0, 3.0];
        let d1 = vec![0.1, 0.0, 0.0];
        let d2 = vec![0.0, 0.2, 0.0];
        let d3 = vec![0.0, 0.0, 0.3];
        let out = multilora_apply(&base, &[d1, d2, d3]);
        assert!((out[0] - 1.1).abs() < 1e-12);
        assert!((out[1] - 2.2).abs() < 1e-12);
        assert!((out[2] - 3.3).abs() < 1e-12);
    }

    #[test]
    fn average_is_deterministic() {
        let a = synthetic_delta(2, 16);
        let b = synthetic_delta(5, 16);
        let m1 = average_merge(&[a.clone(), b.clone()]);
        let m2 = average_merge(&[a, b]);
        assert_eq!(m1, m2);
    }
}
