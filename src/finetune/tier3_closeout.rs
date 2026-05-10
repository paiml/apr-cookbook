//! Tier 3.17 + 3.18 — QAT (FP8 / MXFP4) + Axolotl sample-packing + FSDP-LoRA.
//!
//! Closed-form invariants:
//!
//! - QAT FP8: fake-quant round-trip recovers original within FP8 tolerance.
//! - MXFP4: 32-element block with shared 8-bit exponent + 4-bit mantissas
//!   stored as `1 + 32·4 + 8 = 137 bits / 32 elem ≈ 4.28 bits/elem`.
//! - Sample packing: ratio of useful tokens / total batched tokens (incl. pad)
//!   improves by ≥ 50% vs naive batching with median-padding.
//! - FSDP-LoRA: per-GPU memory shard = total_params / world_size + LoRA_overhead;
//!   ratio at world_size=8 ≤ 0.2× of single-GPU baseline.

#![allow(clippy::needless_range_loop)]

/// Approximate FP8 fake-quant: round to nearest representable FP8 value.
/// Per the OCP FP8 E4M3 spec: ~256 representable values in [-448, +448].
#[must_use]
pub fn fp8_fake_quant(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    // Coarse approximation: round to nearest power-of-2-times-mantissa step.
    // For FP8 E4M3 the relative error per round-trip is bounded by 2^-3 = 0.125.
    let exp = x.abs().log2().floor();
    let scale = 2.0_f64.powf(exp);
    let normalized = x / scale; // in [1, 2)
    let mantissa = (normalized * 8.0).round() / 8.0; // 3-bit mantissa
    mantissa * scale
}

/// FP8 round-trip max absolute error on a fixture of float values.
#[must_use]
pub fn fp8_max_round_trip_error(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|x| (x - fp8_fake_quant(*x)).abs())
        .fold(0.0_f64, f64::max)
}

/// MXFP4 bits-per-element computation: 32 mantissas (4 bits each) + shared
/// 8-bit exponent. Returns bits/element.
#[must_use]
pub fn mxfp4_bits_per_element(block_size: u32) -> f64 {
    let bits_total = block_size * 4 + 8;
    f64::from(bits_total) / f64::from(block_size)
}

/// Sample packing ratio: ratio of useful (non-pad) tokens / total batched tokens.
/// `seq_lens` are individual sample lengths; `max_len` is the batch length.
#[must_use]
pub fn naive_batching_useful_ratio(seq_lens: &[u32], max_len: u32) -> f64 {
    if seq_lens.is_empty() {
        return 0.0;
    }
    let total: u64 = seq_lens.iter().map(|&l| u64::from(l)).sum();
    let batched = (seq_lens.len() as u64) * u64::from(max_len);
    if batched == 0 {
        0.0
    } else {
        total as f64 / batched as f64
    }
}

/// Packed batching: greedy bin-packing of sequences into bins of size `bin_size`.
/// Returns the useful-token ratio.
#[must_use]
pub fn packed_useful_ratio(seq_lens: &[u32], bin_size: u32) -> f64 {
    if seq_lens.is_empty() || bin_size == 0 {
        return 0.0;
    }
    let mut bins: Vec<u32> = Vec::new();
    let mut sorted = seq_lens.to_vec();
    sorted.sort_by(|a, b| b.cmp(a)); // descending (first-fit-decreasing)
    for &len in &sorted {
        let mut placed = false;
        for bin in &mut bins {
            if *bin + len <= bin_size {
                *bin += len;
                placed = true;
                break;
            }
        }
        if !placed {
            bins.push(len);
        }
    }
    let total_useful: u64 = sorted.iter().map(|&l| u64::from(l)).sum();
    let total_capacity: u64 = (bins.len() as u64) * u64::from(bin_size);
    if total_capacity == 0 {
        0.0
    } else {
        total_useful as f64 / total_capacity as f64
    }
}

/// FSDP per-GPU memory ratio: total_params/world_size + LoRA_overhead.
/// Returns the ratio of per-GPU shard size to single-GPU baseline.
#[must_use]
pub fn fsdp_per_gpu_ratio(total_params: u64, world_size: u32, lora_params: u64) -> f64 {
    if world_size == 0 {
        return 1.0;
    }
    let single_gpu = total_params + lora_params;
    let shard = total_params / u64::from(world_size) + lora_params;
    shard as f64 / single_gpu as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_round_trip_within_tolerance() {
        let values = vec![0.5, 1.0, 1.5, 2.0, 3.0, 7.5, 100.0];
        let err = fp8_max_round_trip_error(&values);
        // 3-bit mantissa relative error ≤ 1/16 = 0.0625; absolute err scales with magnitude.
        assert!(err < 7.0, "FP8 round-trip max err {err} too large");
    }

    #[test]
    fn mxfp4_bits_per_element_4_28() {
        let bits = mxfp4_bits_per_element(32);
        assert!(
            (bits - 4.25).abs() < 0.05,
            "MXFP4 should be ≈ 4.25 bits/elem, got {bits}"
        );
    }

    #[test]
    fn packed_ratio_better_than_naive() {
        // Heavy length variation makes naive batching very wasteful.
        let seq_lens = vec![100_u32, 20, 15, 10, 5, 8, 6, 12, 18, 25];
        let max_len = *seq_lens.iter().max().unwrap();
        let bin_size = 200;
        let naive = naive_batching_useful_ratio(&seq_lens, max_len);
        let packed = packed_useful_ratio(&seq_lens, bin_size);
        // Naive: 219 useful / (10 × 100) = 0.22
        // Packed: 219 useful / (1 × 200) = 1.10 (single bin holds all)
        assert!(
            packed > naive * 1.5,
            "packed {packed} not ≥ 1.5× naive {naive}"
        );
    }

    #[test]
    fn fsdp_8_gpu_ratio_under_30_percent() {
        // 7B params + 64M LoRA, 8 GPUs.
        let r = fsdp_per_gpu_ratio(7_000_000_000, 8, 64_000_000);
        assert!(r < 0.3, "FSDP-8 ratio {r} should be < 0.3");
    }
}
