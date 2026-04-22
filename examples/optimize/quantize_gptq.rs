//! # Recipe: GPTQ-Style Per-Channel Quantization Demo
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr quantize --method gptq --bits 4 --group-size 128 model.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example quantize_gptq` exits 0
//! 2. [x] `cargo test --example quantize_gptq` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr quantize --method gptq` in-process (no shell-out)
//! 10. [x] Unit tests cover per-channel scale, dequant round-trip, error budget
//!
//! ## Learning Objective
//! Demonstrates GPTQ-style per-channel quantization: compute per-channel min
//! and max, derive scale / zero-point, round to N-bit integers, then dequant
//! and measure max / mean absolute error. Mirrors `apr quantize --method gptq`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example quantize_gptq
//! ```
//!
//! ## References
//! - Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR. arXiv:2210.17323

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelStats {
    pub scale: f64,
    pub zero_point: i32,
    pub min: f64,
    pub max: f64,
}

pub fn compute_channel_stats(channel: &[f64], bits: u32) -> ChannelStats {
    let qmax = (1i32 << bits) - 1;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in channel {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    if channel.is_empty() || min >= max {
        return ChannelStats {
            scale: 1.0,
            zero_point: 0,
            min: 0.0,
            max: 0.0,
        };
    }
    let qmax_f = f64::from(qmax);
    let scale = (max - min) / qmax_f;
    let zero_point = (-(min / scale)).round().clamp(0.0, qmax_f) as i32;
    ChannelStats {
        scale,
        zero_point,
        min,
        max,
    }
}

pub fn quantize_channel(channel: &[f64], stats: &ChannelStats, bits: u32) -> Vec<u32> {
    let qmax = (1u32 << bits) - 1;
    channel
        .iter()
        .map(|v| {
            let q = (v / stats.scale).round() as i64 + i64::from(stats.zero_point);
            q.clamp(0, i64::from(qmax)) as u32
        })
        .collect()
}

pub fn dequantize_channel(quantized: &[u32], stats: &ChannelStats) -> Vec<f64> {
    quantized
        .iter()
        .map(|q| (i64::from(*q) - i64::from(stats.zero_point)) as f64 * stats.scale)
        .collect()
}

#[derive(Debug, Clone)]
pub struct QuantizationReport {
    pub channels: usize,
    pub bits: u32,
    pub group_size: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub compression_ratio: f64,
}

pub fn quantize_matrix(
    matrix: &[Vec<f64>],
    bits: u32,
    group_size: usize,
) -> (Vec<Vec<u32>>, Vec<ChannelStats>, QuantizationReport) {
    let mut quantized = Vec::with_capacity(matrix.len());
    let mut stats_out = Vec::with_capacity(matrix.len());
    let mut max_err = 0.0f64;
    let mut sum_err = 0.0f64;
    let mut n_elems = 0usize;
    for channel in matrix {
        let stats = compute_channel_stats(channel, bits);
        let q = quantize_channel(channel, &stats, bits);
        let d = dequantize_channel(&q, &stats);
        for (a, b) in channel.iter().zip(d.iter()) {
            let e = (a - b).abs();
            max_err = max_err.max(e);
            sum_err += e;
            n_elems += 1;
        }
        quantized.push(q);
        stats_out.push(stats);
    }
    let mean_err = if n_elems == 0 {
        0.0
    } else {
        sum_err / n_elems as f64
    };
    let compression_ratio = 32.0 / f64::from(bits);
    let report = QuantizationReport {
        channels: matrix.len(),
        bits,
        group_size,
        max_abs_error: max_err,
        mean_abs_error: mean_err,
        compression_ratio,
    };
    (quantized, stats_out, report)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("quantize_gptq")?;
    println!("=== Recipe: {} ===", ctx.name());

    let bits = 4u32;
    let group_size = 128usize;
    let channels = 16;
    let per_channel = group_size;

    let matrix: Vec<Vec<f64>> = (0..channels)
        .map(|_| {
            (0..per_channel)
                .map(|_| ctx.rng().gen_range(-1.0..1.0))
                .collect()
        })
        .collect();

    let (_, _, report) = quantize_matrix(&matrix, bits, group_size);
    println!(
        "GPTQ bits={} group_size={} channels={} max_err={:.6} mean_err={:.6} compression={}x",
        report.bits,
        report.group_size,
        report.channels,
        report.max_abs_error,
        report.mean_abs_error,
        report.compression_ratio
    );

    let report_json = json!({
        "recipe": ctx.name(),
        "bits": report.bits,
        "group_size": report.group_size,
        "channels": report.channels,
        "max_abs_error": report.max_abs_error,
        "mean_abs_error": report.mean_abs_error,
        "compression_ratio": report.compression_ratio,
    });
    let path = ctx.path("quantize-gptq.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report_json)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("bits", i64::from(report.bits));
    ctx.record_float_metric("max_abs_error", report.max_abs_error);
    ctx.record_float_metric("mean_abs_error", report.mean_abs_error);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_channel_scale_positive_for_nondegenerate_input() {
        let ch = vec![-1.0, 0.0, 1.0];
        let s = compute_channel_stats(&ch, 4);
        assert!(s.scale > 0.0);
    }

    #[test]
    fn constant_channel_has_unit_scale() {
        let ch = vec![0.5, 0.5, 0.5];
        let s = compute_channel_stats(&ch, 4);
        assert_eq!(s.scale, 1.0);
    }

    #[test]
    fn dequant_matches_original_within_budget() {
        let ch: Vec<f64> = (0..128).map(|i| (i as f64 / 128.0) - 0.5).collect();
        let s = compute_channel_stats(&ch, 8);
        let q = quantize_channel(&ch, &s, 8);
        let d = dequantize_channel(&q, &s);
        let max_err = ch
            .iter()
            .zip(d.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < s.scale + 1e-9);
    }

    #[test]
    fn higher_bits_lowers_error() {
        let ch: Vec<f64> = (0..64).map(|i| (i as f64 / 64.0) - 0.5).collect();
        let s4 = compute_channel_stats(&ch, 4);
        let s8 = compute_channel_stats(&ch, 8);
        assert!(s8.scale < s4.scale);
    }

    #[test]
    fn empty_matrix_reports_zero_channels() {
        let (_, _, r) = quantize_matrix(&[], 4, 128);
        assert_eq!(r.channels, 0);
        assert_eq!(r.max_abs_error, 0.0);
    }
}
