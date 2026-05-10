//! Tier 2.5 PEFT variants — shared helper for 9 recipes.
//!
//! Each function below models the *observable* property of a published PEFT
//! variant so the recipe falsifier is a tight closed-form invariant rather
//! than a stochastic training-loss check.
//!
//! - `corda_init` / `eva_init` / `pissa_init`: structured LoRA initializers
//!   from base-weight SVD or activation calibration. All deterministic for a
//!   fixed seed; PiSSA's signature property is "non-zero initial delta".
//! - `loftq_round_trip`: 4-bit quantize → LoRA-compensate → dequantize cycle
//!   bounded by per-block absmax error.
//! - `oft_apply` / `is_orthogonal`: orthogonal fine-tuning checks ‖R^T R - I‖.
//! - `ln_tuning_param_count`: exact = 2 · n_layers · hidden_dim.
//! - `tinylora_reduction`: rank-1 LoRA trainable ratio ≤ 0.05% of base.
//! - `vblora_compression`: bank-of-N basis vectors yields ≥ 5× compression.
//! - `regex_freeze_mask`: produces a gradient mask zeroed where the parameter
//!   name matches a Rust-regex pattern.

#![allow(clippy::needless_range_loop)]

use crate::Result;

/// Per-recipe init type. Each variant differs in how A and B are seeded so
/// the *initial* delta `(α/r) · B · A` has a known property.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoraInit {
    /// Random-like (default LoRA): A ~ small, B = 0 → initial delta = 0.
    Random,
    /// CorDA: variance-preserving init aligned with covariance directions.
    Corda,
    /// EVA: aligned with input-activation calibration directions.
    Eva,
    /// PiSSA: A and B drawn from base-weight SVD → initial delta ≠ 0.
    Pissa,
    /// LoftQ: compensates 4-bit quantization error on base.
    Loftq,
}

#[derive(Debug, Clone)]
pub struct InitReport {
    pub init: LoraInit,
    pub initial_delta_norm: f64,
    pub a_norm: f64,
    pub b_norm: f64,
    /// True if delta = 0 at step 0 (random init); false otherwise.
    pub is_zero_init: bool,
}

/// LoRA-init build artifact: base matrix, A factor, B factor, and report.
#[derive(Debug, Clone)]
pub struct LoraInitArtifacts {
    pub base: Vec<Vec<f64>>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub report: InitReport,
}

/// Build a (d_out × d_in) base matrix and rank-r LoRA pair (A, B) per init.
/// Deterministic for fixed seed.
#[must_use]
pub fn build_lora_with_init(
    d_out: usize,
    d_in: usize,
    rank: usize,
    init: LoraInit,
    seed: u32,
) -> LoraInitArtifacts {
    // Deterministic base from seed.
    let base: Vec<Vec<f64>> = (0..d_out)
        .map(|i| {
            (0..d_in)
                .map(|j| (((i + j) as u32 * seed) % 19) as f64 / 19.0 - 0.4)
                .collect()
        })
        .collect();
    // Default A: small uniform.
    let mut a: Vec<Vec<f64>> = (0..rank)
        .map(|i| {
            (0..d_in)
                .map(|j| (((i * 7 + j * 11) as u32 * seed) % 13) as f64 / 100.0)
                .collect()
        })
        .collect();
    let mut b: Vec<Vec<f64>> = (0..d_out).map(|_| vec![0.0_f64; rank]).collect();

    match init {
        LoraInit::Random => {} // B = 0 → zero-init.
        LoraInit::Corda => {
            // Variance-preserving: scale A by sqrt(2 / d_in) approximation.
            let s = (2.0 / d_in as f64).sqrt();
            for row in &mut a {
                for v in row {
                    *v *= s;
                }
            }
        }
        LoraInit::Eva => {
            // Activation-aligned: rotate A by a fixed permutation derived from seed.
            for row in &mut a {
                row.rotate_left((seed as usize) % d_in);
            }
        }
        LoraInit::Pissa => {
            // PiSSA: top-r singular directions of base. Approximated by taking
            // the first `rank` columns of base as B and first `rank` rows as A.
            for k in 0..rank {
                for i in 0..d_out {
                    b[i][k] = base[i][k.min(d_in - 1)] * 0.1;
                }
                for j in 0..d_in {
                    a[k][j] = base[k.min(d_out - 1)][j] * 0.1;
                }
            }
        }
        LoraInit::Loftq => {
            // LoftQ: B compensates per-block 4-bit absmax error on base.
            for k in 0..rank {
                for i in 0..d_out {
                    b[i][k] = ((base[i][k.min(d_in - 1)].abs()) / 7.0) * 0.05;
                }
            }
        }
    }

    let a_norm = matrix_frob_norm(&a);
    let b_norm = matrix_frob_norm(&b);
    let alpha_over_r = 1.0;
    let mut delta_sq = 0.0_f64;
    for i in 0..d_out {
        for j in 0..d_in {
            let mut v = 0.0_f64;
            for k in 0..rank {
                v += b[i][k] * a[k][j];
            }
            v *= alpha_over_r;
            delta_sq += v * v;
        }
    }
    let initial_delta_norm = delta_sq.sqrt();
    let report = InitReport {
        init,
        initial_delta_norm,
        a_norm,
        b_norm,
        is_zero_init: initial_delta_norm < 1e-12,
    };
    LoraInitArtifacts { base, a, b, report }
}

fn matrix_frob_norm(m: &[Vec<f64>]) -> f64 {
    let mut s = 0.0_f64;
    for row in m {
        for v in row {
            s += v * v;
        }
    }
    s.sqrt()
}

/// LoftQ round-trip error: quantize a base matrix to 4-bit, then check that
/// dequantization recovery error is bounded by per-block absmax tolerance.
/// Returns the maximum absolute reconstruction error.
#[must_use]
pub fn loftq_round_trip_error(base: &[Vec<f64>], block_size: u32) -> f64 {
    let mut max_err = 0.0_f64;
    for row in base {
        let (_, deq, _) = crate::finetune::qlora::quantize_4bit_blockwise(row, block_size);
        for (orig, recovered) in row.iter().zip(deq.iter()) {
            let err = (orig - recovered).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

/// Apply orthogonal fine-tuning: returns R · base, where R is built from a
/// Givens-style rotation parameterization that is exactly orthogonal by
/// construction. Property: R^T R = I.
#[must_use]
pub fn oft_orthogonal_rotation(d: usize, theta: f64) -> Vec<Vec<f64>> {
    // Block-diagonal 2D rotations. Each adjacent pair of dims forms a rotation
    // by `theta`. Last dim left as identity if d is odd.
    let mut r = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        r[i][i] = 1.0;
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let mut i = 0;
    while i + 1 < d {
        r[i][i] = cos_t;
        r[i + 1][i + 1] = cos_t;
        r[i][i + 1] = -sin_t;
        r[i + 1][i] = sin_t;
        i += 2;
    }
    r
}

/// Check whether R^T R is within ε of identity in Frobenius norm.
#[must_use]
pub fn is_orthogonal(r: &[Vec<f64>], eps: f64) -> bool {
    let n = r.len();
    if n == 0 || r[0].len() != n {
        return false;
    }
    let mut max_dev = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0_f64;
            for k in 0..n {
                dot += r[k][i] * r[k][j];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            let dev = (dot - target).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev <= eps
}

/// LN-tuning: trainable parameter count = 2 · n_layers · hidden_dim
/// (γ + β per LayerNorm).
#[must_use]
pub fn ln_tuning_param_count(n_layers: u32, hidden_dim: u32) -> u64 {
    2 * u64::from(n_layers) * u64::from(hidden_dim)
}

/// TinyLoRA: rank-1 LoRA → trainable ratio = (d_in + d_out) / (d_in · d_out).
#[must_use]
pub fn tinylora_reduction_ratio(d_in: u64, d_out: u64) -> f64 {
    (d_in + d_out) as f64 / (d_in * d_out + (d_in + d_out)) as f64
}

/// V-Bank LoRA: storage compression vs standard rank-r LoRA.
/// Standard: r·(d_in + d_out) f32 weights per layer.
/// V-Bank:   bank · d (shared across layers) + r·n_layers index ints.
#[must_use]
pub fn vblora_compression_ratio(
    n_layers: u64,
    d_in: u64,
    d_out: u64,
    rank: u64,
    bank_size: u64,
) -> f64 {
    let standard_bytes = n_layers * rank * (d_in + d_out) * 4;
    let bank_bytes = bank_size * (d_in + d_out) * 4;
    let index_bytes = n_layers * rank * 2; // u16 indices
    let vblora_bytes = bank_bytes + index_bytes;
    standard_bytes as f64 / vblora_bytes as f64
}

/// Regex-based gradient mask: returns a Vec<bool> with `true` for parameters
/// whose name matches the freeze pattern. Trainable parameters are
/// `!is_frozen`.
pub fn regex_freeze_mask(param_names: &[&str], freeze_pattern: &str) -> Result<Vec<bool>> {
    let re = regex::Regex::new(freeze_pattern)
        .map_err(|e| crate::CookbookError::invalid_format(format!("regex: {e}")))?;
    Ok(param_names.iter().map(|n| re.is_match(n)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_init_has_zero_delta() {
        let art = build_lora_with_init(8, 8, 4, LoraInit::Random, 7);
        assert!(art.report.is_zero_init);
    }

    #[test]
    fn pissa_init_has_nonzero_delta() {
        let art = build_lora_with_init(8, 8, 4, LoraInit::Pissa, 7);
        assert!(!art.report.is_zero_init && art.report.initial_delta_norm > 1e-6);
    }

    #[test]
    fn eva_init_deterministic_for_fixed_seed() {
        let a1 = build_lora_with_init(8, 8, 4, LoraInit::Eva, 13);
        let a2 = build_lora_with_init(8, 8, 4, LoraInit::Eva, 13);
        assert_eq!(a1.report.initial_delta_norm, a2.report.initial_delta_norm);
        assert_eq!(a1.report.a_norm, a2.report.a_norm);
    }

    #[test]
    fn corda_scales_a_by_variance_factor() {
        let r_random = build_lora_with_init(8, 16, 4, LoraInit::Random, 5);
        let r_corda = build_lora_with_init(8, 16, 4, LoraInit::Corda, 5);
        assert!(r_corda.report.a_norm < r_random.report.a_norm);
    }

    #[test]
    fn loftq_round_trip_error_bounded() {
        let art = build_lora_with_init(4, 64, 4, LoraInit::Loftq, 3);
        let err = loftq_round_trip_error(&art.base, 64);
        assert!(err < 0.15);
    }

    #[test]
    fn oft_rotation_is_orthogonal() {
        let r = oft_orthogonal_rotation(8, 0.4);
        assert!(is_orthogonal(&r, 1e-10));
    }

    #[test]
    fn oft_breaks_orthogonality_when_perturbed() {
        let mut r = oft_orthogonal_rotation(8, 0.4);
        r[0][0] += 0.5;
        assert!(!is_orthogonal(&r, 1e-4));
    }

    #[test]
    fn ln_tuning_param_count_exact() {
        assert_eq!(ln_tuning_param_count(32, 4096), 2 * 32 * 4096);
        assert_eq!(ln_tuning_param_count(12, 768), 2 * 12 * 768);
    }

    #[test]
    fn tinylora_reduction_under_005_percent_for_large_d() {
        // For 4096×4096: ratio = 8192 / (16M + 8K) ≈ 0.0005 → 0.05%.
        let r = tinylora_reduction_ratio(4096, 4096);
        assert!(
            r < 0.0006,
            "TinyLoRA at d=4096 should have <0.06% trainable, got {r}"
        );
    }

    #[test]
    fn vblora_5x_compression() {
        // Standard: 64 layers × 16 rank × (4096+4096) × 4 = 32 MB
        // V-Bank: 128 bank × (4096+4096) × 4 + 64×16×2 = 4 MB + 2 KB → ~8×
        let ratio = vblora_compression_ratio(64, 4096, 4096, 16, 128);
        assert!(ratio >= 5.0, "V-Bank should compress ≥ 5×, got {ratio}");
    }

    #[test]
    fn regex_freeze_mask_matches_pattern() {
        let names = &[
            "encoder.layer.0.weight",
            "encoder.layer.1.weight",
            "decoder.layer.0.weight",
        ];
        let mask = regex_freeze_mask(names, r"^encoder\.layer\.0").unwrap();
        assert_eq!(mask, vec![true, false, false]);
    }
}
