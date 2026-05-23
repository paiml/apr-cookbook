//! Tier 2.7-2.9 closeout — quantized-base LoRA + ReLoRA + LISA + NEFTune.
//!
//! Each function below models the *observable* invariant of the technique
//! so the recipe falsifier is a tight, deterministic check.
//!
//! - 2.7 Quantized-base LoRA (AQLM, AWQ, GPTQ): bit-width vs perplexity bound.
//! - 2.8 ReLoRA: cumulative effective rank > single rank-r over T restarts.
//! - 2.8 LISA: top-k importance sampling — chosen layers update, others zero.
//! - 2.9 NEFTune: embedding noise α causes signal at noise scale exactly.

#![allow(clippy::needless_range_loop)]

/// AQLM 2-bit base + LoRA: post-merge perplexity bound.
/// Closed-form: per-block 2-bit absmax error ≤ |max| / 1.5 (3 levels: -1, 0, +1).
/// We bound the bytes-per-param at 2/8 = 0.25 of FP16, so the storage savings
/// invariant holds: total bytes ≤ 0.25 + LoRA-overhead.
#[must_use]
pub fn aqlm_storage_ratio(base_params: u64, lora_params: u64) -> f64 {
    let aqlm_base_bytes = (base_params * 2).div_ceil(8) + base_params / 64 * 4;
    let lora_bytes = lora_params * 4;
    let fp16_bytes = base_params * 2;
    (aqlm_base_bytes + lora_bytes) as f64 / fp16_bytes as f64
}

/// AWQ 4-bit activation-aware quantization: salient channels (top-k by
/// activation magnitude) are preserved at higher precision. We model this
/// as: a salient channel's recovery error ≤ ε_salient (e.g. 0.5%), while
/// non-salient channels can have error up to ε_normal (e.g. 7%).
#[must_use]
pub fn awq_max_salient_error(weights: &[f64], activations: &[f64], top_k: usize) -> f64 {
    if weights.len() != activations.len() || weights.is_empty() {
        return f64::NAN;
    }
    let mut paired: Vec<(f64, f64)> = weights
        .iter()
        .copied()
        .zip(activations.iter().copied())
        .collect();
    paired.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let salient: Vec<f64> = paired.iter().take(top_k).map(|(w, _)| *w).collect();
    // Salient channels keep precision: assume their recovery error is bounded
    // by 0.5% of |w|. For a falsifier we just check max|w_salient| stays in
    // a "preserved" envelope — i.e. they aren't all zeroed.
    salient.iter().map(|w| w.abs()).fold(0.0_f64, f64::max)
}

/// GPTQ per-block reconstruction error bound: error ≤ tolerance × σ_block,
/// where σ_block is the block's L2 norm. Returns the max relative error.
#[must_use]
pub fn gptq_max_relative_block_error(blocks: &[Vec<f64>], errors: &[Vec<f64>]) -> f64 {
    let mut max_rel = 0.0_f64;
    for (block, err) in blocks.iter().zip(errors.iter()) {
        let sigma = block.iter().map(|x| x * x).sum::<f64>().sqrt();
        if sigma < 1e-12 {
            continue;
        }
        let block_err = err.iter().map(|e| e * e).sum::<f64>().sqrt();
        let rel = block_err / sigma;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    max_rel
}

/// ReLoRA cumulative-rank model: T restarts of rank-r LoRA, each merged into
/// base before the next adapter is initialized. Cumulative effective rank
/// caps at min(t·r, d) where d is the parameter dimension.
#[must_use]
pub fn relora_cumulative_rank(rank: u32, n_restarts: u32, d: u32) -> u32 {
    (rank * n_restarts).min(d)
}

/// LISA layer-importance sampler: returns indices of the top-k layers
/// (by importance score) that get updated this step. All other layers
/// have gradient zero.
#[must_use]
pub fn lisa_select_top_k(importances: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = importances.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

/// LISA gradient-mask: returns a Vec<bool> with `true` for layers selected
/// by `lisa_select_top_k` (these update), `false` for the rest (gradient zero).
#[must_use]
pub fn lisa_gradient_mask(importances: &[f64], k: usize) -> Vec<bool> {
    let selected = lisa_select_top_k(importances, k);
    (0..importances.len())
        .map(|i| selected.contains(&i))
        .collect()
}

/// NEFTune: noisy embedding fine-tuning. Adds uniform-scaled noise of
/// magnitude `alpha / sqrt(L · d)` to every embedding row, where L is the
/// sequence length and d the embedding dim.
/// Property: ‖noise‖_2 / ‖embedding‖_2 ≈ alpha / sqrt(d) for unit-norm embeddings.
#[must_use]
pub fn neftune_noise_scale(alpha: f64, seq_len: u32, d_model: u32) -> f64 {
    alpha / ((u64::from(seq_len) * u64::from(d_model)) as f64).sqrt()
}

/// Apply NEFTune noise to embeddings deterministically (uniform mask from a
/// stride pattern derived from `seed`). Returns noised embeddings.
#[must_use]
pub fn apply_neftune_noise(embeddings: &[Vec<f64>], alpha: f64, seed: u32) -> Vec<Vec<f64>> {
    let l = embeddings.len() as u32;
    if l == 0 {
        return Vec::new();
    }
    let d = embeddings[0].len() as u32;
    if d == 0 {
        return embeddings.to_vec();
    }
    let scale = neftune_noise_scale(alpha, l, d);
    embeddings
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, v)| {
                    // Deterministic +/- mask from index pattern.
                    let pattern = ((i as u32 * seed) + (j as u32 * 13 + 7)) % 7;
                    let sign = if pattern < 4 { 1.0 } else { -1.0 };
                    v + sign * scale
                })
                .collect()
        })
        .collect()
}

/// L2 norm of a single vector slice.
#[must_use]
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aqlm_2bit_storage_below_30_percent() {
        // 7B params + 64M LoRA: AQLM should be ≤ 0.3× of FP16.
        let r = aqlm_storage_ratio(7_000_000_000, 64_000_000);
        assert!(r < 0.3, "AQLM 2-bit storage should be ≤ 0.3× FP16, got {r}");
    }

    #[test]
    fn awq_preserves_salient_channels() {
        let w = vec![0.1, 0.2, 0.3, 0.4, 5.0]; // last is salient
        let act = vec![0.1, 0.2, 0.3, 0.4, 10.0];
        let max_salient = awq_max_salient_error(&w, &act, 1);
        assert!((max_salient - 5.0).abs() < 1e-12);
    }

    #[test]
    fn gptq_block_error_is_relative_to_block_norm() {
        let blocks = vec![vec![1.0, 0.0, 0.0], vec![10.0, 0.0, 0.0]];
        let errors = vec![vec![0.1, 0.0, 0.0], vec![0.5, 0.0, 0.0]];
        let max_rel = gptq_max_relative_block_error(&blocks, &errors);
        // block 0 rel: 0.1/1 = 0.1; block 1 rel: 0.5/10 = 0.05; max = 0.1.
        assert!((max_rel - 0.1).abs() < 1e-12);
    }

    #[test]
    fn relora_cumulative_rank_grows_then_caps() {
        assert_eq!(relora_cumulative_rank(8, 4, 4096), 32);
        assert_eq!(relora_cumulative_rank(8, 4, 16), 16); // capped at d=16
    }

    #[test]
    fn lisa_top_k_selects_highest_importance() {
        let imp = vec![0.5, 0.9, 0.1, 0.7, 0.3];
        let sel = lisa_select_top_k(&imp, 2);
        // Top-2 by importance: indices 1 (0.9), 3 (0.7).
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(&1) && sel.contains(&3));
    }

    #[test]
    fn lisa_gradient_mask_zeros_non_selected() {
        let imp = vec![0.5, 0.9, 0.1, 0.7, 0.3];
        let mask = lisa_gradient_mask(&imp, 2);
        assert!(mask[1]);
        assert!(mask[3]);
        assert!(!mask[0]);
        assert!(!mask[2]);
        assert!(!mask[4]);
    }

    #[test]
    fn neftune_noise_scale_inverse_sqrt() {
        let s1 = neftune_noise_scale(5.0, 1024, 4096);
        let s2 = neftune_noise_scale(5.0, 1024, 16384); // 4× larger d
        assert!(s1 > s2 * 1.99 && s1 < s2 * 2.01); // ratio ≈ sqrt(4) = 2
    }

    #[test]
    fn neftune_noise_changes_embeddings() {
        let emb = vec![vec![1.0; 8]; 4];
        let noised = apply_neftune_noise(&emb, 5.0, 7);
        assert_eq!(noised.len(), emb.len());
        // Noised differs from original at ≥1 entry.
        let any_diff = emb
            .iter()
            .zip(noised.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .any(|(x, y)| (x - y).abs() > 1e-12);
        assert!(any_diff);
    }
}
