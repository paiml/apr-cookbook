//! Tier 2.2 QLoRA — shared helper.
//!
//! QLoRA = 4-bit quantized frozen base + full-precision LoRA adapter.
//! Memory: ~0.3× of FP16 base + LoRA when fully quantized to 4-bit.
//!
//! This helper models the *math* of QLoRA without depending on actual
//! quantization kernels. Per the QLoRA paper:
//!   - 4-bit absmax-quantization: q = round(x / scale * 7), recovered as
//!     dequant(q) = (q / 7) * scale
//!   - Per-block scaling (block_size = 64): scale = max|block|
//!   - Double-quantization: scales themselves quantized to 8-bit fp,
//!     reducing scale-storage memory by ~30%.

#[derive(Debug, Clone, PartialEq)]
pub struct QuantStats {
    pub original_bits: u32,
    pub quantized_bits: u32,
    pub n_blocks: u32,
    pub block_size: u32,
    pub double_quant: bool,
    pub scale_storage_bits_per_block: u32,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
}

impl QuantStats {
    /// Memory ratio: quantized / original.
    /// At 4-bit + 8-bit double-quant scales, expect ~0.3× of FP16.
    #[must_use]
    pub fn memory_ratio_vs_fp16(&self) -> f64 {
        let n_params = self.n_blocks as f64 * self.block_size as f64;
        let quantized_bits = n_params * self.quantized_bits as f64
            + self.n_blocks as f64 * self.scale_storage_bits_per_block as f64;
        let fp16_bits = n_params * 16.0;
        quantized_bits / fp16_bits
    }
}

/// Quantize a vector to 4-bit absmax with per-block scaling.
/// Returns (quantized indices in [-7, 7], dequantized recovery).
#[must_use]
pub fn quantize_4bit_blockwise(x: &[f64], block_size: u32) -> (Vec<i8>, Vec<f64>, QuantStats) {
    let block_size_us = block_size as usize;
    let n = x.len();
    let n_blocks = n.div_ceil(block_size_us);
    let mut q = Vec::with_capacity(n);
    let mut deq = Vec::with_capacity(n);
    let mut max_err = 0.0_f64;
    let mut sum_err = 0.0_f64;
    for b in 0..n_blocks {
        let start = b * block_size_us;
        let end = (start + block_size_us).min(n);
        let block = &x[start..end];
        let scale = block
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-8);
        for &v in block {
            let qi = ((v / scale) * 7.0).round() as i8;
            let qi = qi.clamp(-7, 7);
            let dq = (f64::from(qi) / 7.0) * scale;
            let err = (v - dq).abs();
            max_err = max_err.max(err);
            sum_err += err;
            q.push(qi);
            deq.push(dq);
        }
    }
    let stats = QuantStats {
        original_bits: 16,
        quantized_bits: 4,
        n_blocks: n_blocks as u32,
        block_size,
        double_quant: false,
        scale_storage_bits_per_block: 32, // f32 scale per block
        max_abs_error: max_err,
        mean_abs_error: sum_err / n as f64,
    };
    (q, deq, stats)
}

/// Apply double-quantization (8-bit on scales) to halve scale-storage cost.
#[must_use]
pub fn enable_double_quant(stats: &QuantStats) -> QuantStats {
    QuantStats {
        double_quant: true,
        // 8-bit scale + 8-bit second-level scale on every 256-block group
        // gives ~10 bits/block effective vs 32 originally.
        scale_storage_bits_per_block: 10,
        ..stats.clone()
    }
}

/// QLoRA memory model: combined 4-bit base + FP32 LoRA.
#[derive(Debug, Clone, PartialEq)]
pub struct QloraMemoryReport {
    pub base_4bit_mb: f64,
    pub lora_fp32_mb: f64,
    pub total_mb: f64,
    pub fp16_baseline_mb: f64,
    pub savings_ratio: f64,
}

#[must_use]
pub fn qlora_memory(base_params: u64, lora_trainable: u64) -> QloraMemoryReport {
    let base_4bit_bytes = (base_params * 4).div_ceil(8) + base_params / 64 * 4; // 4-bit + per-block scale
    let lora_fp32_bytes = lora_trainable * 4;
    let fp16_bytes = base_params * 2;
    let total = base_4bit_bytes + lora_fp32_bytes;
    QloraMemoryReport {
        base_4bit_mb: base_4bit_bytes as f64 / 1_048_576.0,
        lora_fp32_mb: lora_fp32_bytes as f64 / 1_048_576.0,
        total_mb: total as f64 / 1_048_576.0,
        fp16_baseline_mb: fp16_bytes as f64 / 1_048_576.0,
        savings_ratio: total as f64 / fp16_bytes as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_recovers_signal_within_tolerance() {
        let x: Vec<f64> = (0..256).map(|i| (i as f64).sin()).collect();
        let (_, dq, stats) = quantize_4bit_blockwise(&x, 64);
        // Per-block 4-bit absmax: max relative error per block ≤ 1/14 ≈ 7.1%
        // Combined with sinusoidal signal: max abs error well under 0.1
        assert!(
            stats.max_abs_error < 0.15,
            "max abs error {} should be < 0.15",
            stats.max_abs_error
        );
        assert_eq!(dq.len(), x.len());
    }

    #[test]
    fn double_quant_reduces_scale_storage() {
        let (_, _, stats) = quantize_4bit_blockwise(&[1.0; 256], 64);
        let double = enable_double_quant(&stats);
        assert!(
            double.scale_storage_bits_per_block < stats.scale_storage_bits_per_block,
            "double-quant should reduce scale storage: {} → {}",
            stats.scale_storage_bits_per_block,
            double.scale_storage_bits_per_block
        );
    }

    #[test]
    fn memory_ratio_around_30_percent_for_4bit() {
        let big = vec![1.0_f64; 4096];
        let (_, _, stats) = quantize_4bit_blockwise(&big, 64);
        let ratio = stats.memory_ratio_vs_fp16();
        // 4-bit + f32 scale per 64-block: (4 + 32/64) / 16 = 4.5/16 = 0.28
        assert!(
            ratio > 0.25 && ratio < 0.35,
            "4-bit memory ratio should be ~0.3 of fp16, got {ratio}"
        );
    }

    #[test]
    fn double_quant_further_reduces_memory() {
        let big = vec![1.0_f64; 4096];
        let (_, _, stats) = quantize_4bit_blockwise(&big, 64);
        let double = enable_double_quant(&stats);
        assert!(double.memory_ratio_vs_fp16() < stats.memory_ratio_vs_fp16());
    }

    #[test]
    fn qlora_memory_report() {
        // 7B params + 64M LoRA trainable
        let report = qlora_memory(7_000_000_000, 64_000_000);
        assert!(
            report.savings_ratio < 0.4,
            "QLoRA should be ≤ 0.4× FP16 baseline"
        );
        assert!(report.total_mb < report.fp16_baseline_mb);
    }

    #[test]
    fn deterministic_quantization() {
        let x: Vec<f64> = (0..128).map(|i| (i as f64) * 0.01).collect();
        let (q1, _, _) = quantize_4bit_blockwise(&x, 64);
        let (q2, _, _) = quantize_4bit_blockwise(&x, 64);
        assert_eq!(q1, q2);
    }
}
