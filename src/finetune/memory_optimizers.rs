//! Tier 2.6 Memory-efficient optimizers — shared helper for 5 recipes.
//!
//! Models the *observable memory and update properties* of the optimizers
//! (GaLore, BAdam, Apollo, DoRA, freeze-tuning) as closed-form formulas so
//! recipe falsifiers are deterministic invariants instead of stochastic
//! training-loss claims.
//!
//! - GaLore: low-rank gradient projection saves optimizer state by `1 − r/d`.
//! - BAdam: only one block is active per step → per-step state ≤ 1/n_blocks.
//! - Apollo: memory ≈ momentum + low-rank approx; ratio ≤ 0.6× of AdamW.
//! - DoRA: trained ΔW decomposes as magnitude × direction.
//! - Freeze-tuning: gradient mask is zero on frozen layers, nonzero otherwise.

#![allow(clippy::needless_range_loop)]

/// Bytes-per-parameter for FP32 momentum + variance in standard Adam.
const ADAM_BYTES_PER_PARAM: f64 = 8.0;

/// GaLore optimizer-state memory for a (d_out × d_in) projection at rank r.
///
/// Standard Adam holds momentum + variance for ALL d_out·d_in params (8 bytes).
/// GaLore projects gradients into a rank-r subspace: only r·(d_in + d_out)
/// state vectors per layer. Memory ratio = (r/d) for d ≫ r.
#[must_use]
pub fn galore_memory_ratio(d_out: u64, d_in: u64, rank: u64) -> f64 {
    let standard = (d_out * d_in) as f64 * ADAM_BYTES_PER_PARAM;
    let galore = (rank * (d_out + d_in)) as f64 * ADAM_BYTES_PER_PARAM;
    galore / standard
}

/// BAdam: block-wise Adam updates one of `n_blocks` parameter blocks per step.
/// At any moment, optimizer state is held only for the active block, so
/// peak memory = total / n_blocks.
#[must_use]
pub fn badam_peak_memory_ratio(n_blocks: u32) -> f64 {
    1.0 / f64::from(n_blocks)
}

/// BAdam invariant: parameter mass within an inactive block is conserved
/// across an optimizer step (only the active block updates).
#[must_use]
pub fn block_mass_after_update(
    params_before: &[f64],
    update: &[f64],
    block_starts: &[usize],
    active_block: usize,
) -> Vec<f64> {
    if params_before.len() != update.len() || active_block >= block_starts.len() {
        return params_before.to_vec();
    }
    let mut params = params_before.to_vec();
    let n = params.len();
    let start = block_starts[active_block];
    let end = block_starts.get(active_block + 1).copied().unwrap_or(n);
    for i in start..end.min(n) {
        params[i] += update[i];
    }
    params
}

/// L1 mass of a slice.
#[must_use]
pub fn l1_mass(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// Apollo: memory ratio of low-memory optimizer vs AdamW.
/// Apollo holds 1× momentum + low-rank variance approximation.
/// Ratio ≈ (1 + r/d) / 2 for d ≫ r → ~0.5 to 0.6.
#[must_use]
pub fn apollo_memory_ratio(d_out: u64, d_in: u64, rank: u64) -> f64 {
    let standard = (d_out * d_in) as f64 * ADAM_BYTES_PER_PARAM;
    let momentum = (d_out * d_in) as f64 * 4.0; // f32 momentum
    let lr_variance = (rank * (d_out + d_in)) as f64 * 4.0;
    (momentum + lr_variance) / standard
}

/// DoRA decomposition: weight = magnitude · (direction).
/// Property: |direction| = 1 (unit-norm), and weight = magnitude × direction
/// recovers original weight. Returns (magnitude, direction) so the recipe
/// asserts ‖direction‖₂ = 1 and reconstruction holds.
#[must_use]
pub fn dora_decompose(weight: &[f64]) -> (f64, Vec<f64>) {
    let magnitude = weight.iter().map(|v| v * v).sum::<f64>().sqrt();
    if magnitude < 1e-12 {
        return (0.0, vec![0.0; weight.len()]);
    }
    let direction = weight.iter().map(|v| v / magnitude).collect();
    (magnitude, direction)
}

#[must_use]
pub fn dora_reconstruct(magnitude: f64, direction: &[f64]) -> Vec<f64> {
    direction.iter().map(|v| magnitude * v).collect()
}

#[must_use]
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Freeze-tuning: build a gradient mask given a list of layer-name prefixes
/// to freeze. Returns Vec<bool>: true if frozen (gradient zero), false if
/// trainable.
#[must_use]
pub fn freeze_mask(layer_names: &[&str], freeze_prefixes: &[&str]) -> Vec<bool> {
    layer_names
        .iter()
        .map(|n| freeze_prefixes.iter().any(|p| n.starts_with(p)))
        .collect()
}

/// Apply gradient mask to a slice of gradients.
#[must_use]
pub fn apply_freeze_mask(gradients: &[f64], mask: &[bool]) -> Vec<f64> {
    gradients
        .iter()
        .zip(mask.iter())
        .map(|(g, &frozen)| if frozen { 0.0 } else { *g })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galore_memory_under_50_percent_at_low_rank() {
        // 4096×4096 with rank 128: ratio = 128 × (4096+4096) / (4096^2) ≈ 0.063
        let ratio = galore_memory_ratio(4096, 4096, 128);
        assert!(
            ratio < 0.5,
            "GaLore at low rank must be ≤ 0.5× of Adam, got {ratio}"
        );
    }

    #[test]
    fn badam_peak_memory_inversely_proportional() {
        assert_eq!(badam_peak_memory_ratio(8), 0.125);
        assert_eq!(badam_peak_memory_ratio(16), 0.0625);
    }

    #[test]
    fn badam_inactive_block_mass_preserved() {
        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let update = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let starts = vec![0, 2, 4]; // blocks: [0..2], [2..4], [4..6]
        let updated = block_mass_after_update(&params, &update, &starts, 1);
        // Only block 1 (indices 2..4) updates.
        assert_eq!(updated[0], 1.0); // unchanged
        assert_eq!(updated[1], 2.0); // unchanged
        assert!((updated[2] - 3.1).abs() < 1e-12);
        assert!((updated[3] - 4.1).abs() < 1e-12);
        assert_eq!(updated[4], 5.0); // unchanged
        assert_eq!(updated[5], 6.0); // unchanged
                                     // Inactive block 0 mass unchanged.
        assert_eq!(l1_mass(&updated[..2]), l1_mass(&params[..2]));
    }

    #[test]
    fn apollo_memory_under_60_percent() {
        let ratio = apollo_memory_ratio(4096, 4096, 64);
        assert!(
            ratio < 0.6,
            "Apollo at rank-64 must be ≤ 0.6× of AdamW, got {ratio}"
        );
    }

    #[test]
    fn dora_round_trip_recovers_weight() {
        let w = vec![1.0, 2.0, 3.0, -4.0, 5.0];
        let (m, d) = dora_decompose(&w);
        assert!((vec_norm(&d) - 1.0).abs() < 1e-12);
        let r = dora_reconstruct(m, &d);
        for (a, b) in w.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn dora_direction_has_unit_norm() {
        let w = vec![1.0; 16];
        let (_, d) = dora_decompose(&w);
        assert!((vec_norm(&d) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn freeze_mask_zeros_frozen_gradients() {
        let layers = &["embed.weight", "layer.0.q", "layer.0.k", "head.weight"];
        let mask = freeze_mask(layers, &["embed.", "head."]);
        assert_eq!(mask, vec![true, false, false, true]);
        let grads = vec![0.5, 0.3, 0.4, 0.7];
        let masked = apply_freeze_mask(&grads, &mask);
        assert_eq!(masked, vec![0.0, 0.3, 0.4, 0.0]);
    }
}
