//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::f32;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantFormat {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_1,
}
impl QuantFormat {
    #[must_use]
    pub const fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::Q4_0 => 4.5,
            Self::Q4_1 => 5.0,
        }
    }
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_weight()
    }
    #[must_use]
    pub const fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q4_0 | Self::Q4_1 => 32,
        }
    }
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
        }
    }
    pub const ALL: [Self; 6] = [
        Self::F32,
        Self::F16,
        Self::BF16,
        Self::Q8_0,
        Self::Q4_0,
        Self::Q4_1,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationResult {
    pub original_format: QuantFormat,
    pub target_format: QuantFormat,
    pub weight_count: usize,
    pub original_size_bytes: usize,
    pub quantized_size_bytes: usize,
    pub compression_ratio: f32,
    pub mse: f64,
    pub snr_db: f64,
    pub psnr_db: f64,
    pub max_abs_error: f32,
    pub changed_pct: f32,
    pub time_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantAnalysis {
    pub model_name: String,
    pub total_params: usize,
    pub results: Vec<QuantizationResult>,
    pub recommended_format: QuantFormat,
    pub recommendation_reason: String,
}

pub fn quantize_and_measure(weights: &[f32], target: QuantFormat) -> Result<QuantizationResult> {
    let start = Instant::now();
    let quantized = quantize_weights(weights, target)?;
    let dequantized = dequantize_weights(&quantized, target)?;
    let mse = compute_mse(weights, &dequantized);
    let snr_db = compute_snr_db(weights, &dequantized);
    let psnr_db = compute_psnr_db(weights, &dequantized);
    let max_abs_error = weights
        .iter()
        .zip(dequantized.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let changed_pct = if weights.is_empty() {
        0.0
    } else {
        weights
            .iter()
            .zip(dequantized.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-6)
            .count() as f32
            / weights.len() as f32
            * 100.0
    };
    let orig_sz = weights.len() * 4;
    let quant_sz = estimate_quantized_size(weights.len(), target);
    Ok(QuantizationResult {
        original_format: QuantFormat::F32,
        target_format: target,
        weight_count: weights.len(),
        original_size_bytes: orig_sz,
        quantized_size_bytes: quant_sz,
        compression_ratio: orig_sz as f32 / quant_sz as f32,
        mse,
        snr_db,
        psnr_db,
        max_abs_error,
        changed_pct,
        time_us: start.elapsed().as_micros() as u64,
    })
}

/// Compute scale and its inverse from an absolute maximum, with a fallback for zero.
pub fn scale_and_inv(abs_max: f32, divisor: f32) -> (f32, f32) {
    let scale = if abs_max > 0.0 {
        abs_max / divisor
    } else {
        1.0
    };
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    (scale, inv)
}

/// Look up a quantized nibble value from a block, returning `default` for out-of-bounds indices.
pub fn nibble_val(
    block: &[f32],
    idx: usize,
    inv: f32,
    offset: f32,
    lo: f32,
    hi: f32,
    default: u8,
) -> u8 {
    if idx < block.len() {
        ((block[idx] - offset) * inv).round().clamp(lo, hi) as i8 as u8
    } else {
        default
    }
}

/// Pack pairs of nibble-quantized values from a block into bytes.
pub fn pack_nibble_pairs(
    block: &[f32],
    inv: f32,
    offset: f32,
    lo: f32,
    hi: f32,
    bias: u8,
    default: u8,
    bytes: &mut Vec<u8>,
) {
    let mut i = 0;
    while i < 32 {
        let q0 = nibble_val(block, i, inv, offset, lo, hi, default).wrapping_add(bias);
        let q1 = nibble_val(block, i + 1, inv, offset, lo, hi, default).wrapping_add(bias);
        bytes.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        i += 2;
    }
}

/// Quantize a Q8_0 block: 4-byte scale + 32 signed 8-bit values.
pub fn quantize_q8_block(block: &[f32], bytes: &mut Vec<u8>) {
    let abs_max = block.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
    let (scale, inv) = scale_and_inv(abs_max, 127.0);
    bytes.extend_from_slice(&scale.to_le_bytes());
    for &w in block {
        bytes.push((w * inv).round().clamp(-128.0, 127.0) as i8 as u8);
    }
    for _ in block.len()..32 {
        bytes.push(0);
    }
}

/// Quantize a Q4_0 block: 4-byte scale + 16 packed nibble pairs (symmetric around zero).
pub fn quantize_q4_0_block(block: &[f32], bytes: &mut Vec<u8>) {
    let abs_max = block.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
    let (scale, inv) = scale_and_inv(abs_max, 7.0);
    bytes.extend_from_slice(&scale.to_le_bytes());
    pack_nibble_pairs(block, inv, 0.0, -8.0, 7.0, 8, 8, bytes);
}

/// Quantize a Q4_1 block: 4-byte scale + 4-byte min + 16 packed nibble pairs (asymmetric).
pub fn quantize_q4_1_block(block: &[f32], bytes: &mut Vec<u8>) {
    let min = block.iter().copied().fold(f32::INFINITY, f32::min);
    let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let (scale, inv) = scale_and_inv(range, 15.0);
    bytes.extend_from_slice(&scale.to_le_bytes());
    bytes.extend_from_slice(&min.to_le_bytes());
    pack_nibble_pairs(block, inv, min, 0.0, 15.0, 0, 0, bytes);
}

pub fn quantize_weights(weights: &[f32], target: QuantFormat) -> Result<Vec<u8>> {
    match target {
        QuantFormat::F32 => Ok(weights.iter().flat_map(|&f| f.to_le_bytes()).collect()),
        QuantFormat::F16 => Ok(weights
            .iter()
            .flat_map(|&w| f32_to_f16_bits(w).to_le_bytes())
            .collect()),
        QuantFormat::BF16 => Ok(weights
            .iter()
            .flat_map(|&w| ((w.to_bits() >> 16) as u16).to_le_bytes())
            .collect()),
        QuantFormat::Q8_0 => quantize_block(weights, 32, quantize_q8_block),
        QuantFormat::Q4_0 => quantize_block(weights, 32, quantize_q4_0_block),
        QuantFormat::Q4_1 => quantize_block(weights, 32, quantize_q4_1_block),
    }
}

pub fn quantize_block(
    weights: &[f32],
    bs: usize,
    f: impl Fn(&[f32], &mut Vec<u8>),
) -> Result<Vec<u8>> {
    let nb = weights.len().div_ceil(bs);
    let mut bytes = Vec::new();
    for bi in 0..nb {
        let start = bi * bs;
        f(&weights[start..(start + bs).min(weights.len())], &mut bytes);
    }
    Ok(bytes)
}

pub fn dequantize_weights(data: &[u8], format: QuantFormat) -> Result<Vec<f32>> {
    match format {
        QuantFormat::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        QuantFormat::F16 => Ok(data
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        QuantFormat::BF16 => Ok(data
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        QuantFormat::Q8_0 => {
            let bs = 36;
            let nb = data.len() / bs;
            let mut w = Vec::with_capacity(nb * 32);
            for bi in 0..nb {
                let o = bi * bs;
                let s = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                for i in 0..32 {
                    w.push(f32::from(data[o + 4 + i] as i8) * s);
                }
            }
            Ok(w)
        }
        QuantFormat::Q4_0 => {
            let bs = 20;
            let nb = data.len() / bs;
            let mut w = Vec::with_capacity(nb * 32);
            for bi in 0..nb {
                let o = bi * bs;
                let s = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                for i in 0..16 {
                    let p = data[o + 4 + i];
                    w.push(f32::from((p & 0x0F) as i8 - 8) * s);
                    w.push(f32::from(((p >> 4) & 0x0F) as i8 - 8) * s);
                }
            }
            Ok(w)
        }
        QuantFormat::Q4_1 => {
            let bs = 24;
            let nb = data.len() / bs;
            let mut w = Vec::with_capacity(nb * 32);
            for bi in 0..nb {
                let o = bi * bs;
                let s = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                let mn = f32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]);
                for i in 0..16 {
                    let p = data[o + 8 + i];
                    w.push(f32::from(p & 0x0F) * s + mn);
                    w.push(f32::from((p >> 4) & 0x0F) * s + mn);
                }
            }
            Ok(w)
        }
    }
}

pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 255 {
        return if frac == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7C00 | ((frac >> 13) as u16).max(1)
        };
    }
    if exp == 0 {
        return sign << 15;
    }
    let ne = exp - 127 + 15;
    if ne >= 31 {
        return (sign << 15) | 0x7C00;
    } else if ne <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((ne as u16) << 10) | ((frac >> 13) as u16)
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = frac;
        let mut e = -14_i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        return f32::from_bits((sign << 31) | ((e + 127) as u32) << 23 | (m << 13));
    }
    if exp == 31 {
        return if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000 | (frac << 13))
        };
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

pub fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}
pub fn compute_snr_db(a: &[f32], b: &[f32]) -> f64 {
    let sp: f64 = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let np: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    if np < 1e-30 {
        100.0
    } else {
        10.0 * (sp / np).log10()
    }
}
pub fn compute_psnr_db(a: &[f32], b: &[f32]) -> f64 {
    let mx: f64 = a.iter().map(|&x| f64::from(x).abs()).fold(0.0, f64::max);
    let mse = compute_mse(a, b);
    if mse < 1e-30 {
        100.0
    } else {
        10.0 * ((mx * mx) / mse).log10()
    }
}
pub fn estimate_quantized_size(n: usize, fmt: QuantFormat) -> usize {
    match fmt {
        QuantFormat::F32 => n * 4,
        QuantFormat::F16 | QuantFormat::BF16 => n * 2,
        QuantFormat::Q8_0 => n.div_ceil(32) * 36,
        QuantFormat::Q4_0 => n.div_ceil(32) * 20,
        QuantFormat::Q4_1 => n.div_ceil(32) * 24,
    }
}

pub fn analyze_quantization(model_name: &str, weights: &[f32]) -> Result<QuantAnalysis> {
    let results: Vec<QuantizationResult> = QuantFormat::ALL
        .iter()
        .map(|&f| quantize_and_measure(weights, f))
        .collect::<Result<_>>()?;
    let (rec, reason) = recommend_format(&results);
    Ok(QuantAnalysis {
        model_name: model_name.into(),
        total_params: weights.len(),
        results,
        recommended_format: rec,
        recommendation_reason: reason,
    })
}

pub fn recommend_format(results: &[QuantizationResult]) -> (QuantFormat, String) {
    let mut best = (f64::NEG_INFINITY, QuantFormat::Q8_0, String::new());
    for r in results {
        if r.target_format == QuantFormat::F32 {
            continue;
        }
        let score = (r.snr_db / 50.0).min(1.0) + (f64::from(r.compression_ratio) / 8.0).min(1.0);
        if score > best.0 {
            best = (
                score,
                r.target_format,
                format!(
                    "Best tradeoff: {:.1} dB SNR, {:.2}x compression",
                    r.snr_db, r.compression_ratio
                ),
            );
        }
    }
    (best.1, best.2)
}
