//! # Recipe: Decode Temperature Sweep
//!
//! **Category**: inference
//! **CLI Equivalent**: `apr run model.apr --prompt "..." --temperature <T>`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example inference_run_temperature_sweep` exits 0
//! 2. [x] `cargo test --example inference_run_temperature_sweep` passes
//! 3. [x] Deterministic output (same seed → same sequences)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] T=0.0 is pure argmax (always picks top token)
//! 8. [x] Entropy is monotone non-decreasing in T across the sweep
//!
//! ## Learning Objective
//! Demonstrates the *decode temperature sweep* pattern behind `apr run`.
//! Starting from a fixed 8-token logit distribution, we softmax-sample at
//! T ∈ {0.0, 0.3, 0.7, 1.0, 1.5} and trace the output shift from deterministic
//! argmax (T=0.0) to high-entropy divergence (T=1.5). Reveals the trade-off
//! Holtzman et al. catalogue as "neural text degeneration" — too low and you
//! get repetition; too high and you get nonsense.
//!
//! ## Run Command
//! ```bash
//! cargo run --example inference_run_temperature_sweep
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          --temperature 0.7  # APR native
//! apr run model.gguf         --temperature 0.7  # GGUF
//! apr run model.safetensors  --temperature 0.7  # HF SafeTensors
//! ```
//!
//! ## References
//! - Holtzman, A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. arXiv:1904.09751

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;

/// A deterministic 8-slot logit distribution ordered high → low.
/// Token index 0 is the argmax; index 7 is the tail.
#[must_use]
pub fn canonical_logits() -> Vec<f32> {
    vec![4.0, 3.6, 3.0, 2.4, 1.8, 1.0, 0.2, -0.5]
}

/// Temperature-scaled softmax. T=0.0 returns a one-hot on argmax.
#[must_use]
pub fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 {
        // Pure argmax — one-hot.
        let argmax = argmax_index(logits);
        let mut out = vec![0.0; logits.len()];
        if argmax < out.len() {
            out[argmax] = 1.0;
        }
        return out;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| ((v - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

#[must_use]
pub fn argmax_index(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    best
}

/// Sample a token index from a probability distribution using `rng`.
///
/// For T=0.0 the distribution is already one-hot, so we fall through to
/// argmax without consuming RNG state.
pub fn sample_token(probs: &[f32], rng: &mut impl Rng) -> usize {
    if probs.iter().filter(|p| **p > 0.0).count() <= 1 {
        return argmax_index(probs);
    }
    let r: f32 = rng.gen();
    let mut acc = 0.0_f32;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Shannon entropy (nats).
#[must_use]
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Sweep outcome at a single temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepRow {
    pub temperature: f32,
    pub entropy: f32,
    pub sampled_tokens: Vec<usize>,
    pub unique_count: usize,
}

/// Sample `k` tokens from the logit distribution at temperature `t`.
pub fn sweep_at(logits: &[f32], temperature: f32, samples: usize, rng: &mut impl Rng) -> SweepRow {
    let probs = softmax_temperature(logits, temperature);
    let mut tokens = Vec::with_capacity(samples);
    for _ in 0..samples {
        tokens.push(sample_token(&probs, rng));
    }
    let unique: std::collections::HashSet<_> = tokens.iter().copied().collect();
    SweepRow {
        temperature,
        entropy: entropy(&probs),
        sampled_tokens: tokens,
        unique_count: unique.len(),
    }
}

/// Render the sweep as an aligned table.
#[must_use]
pub fn render_sweep(rows: &[SweepRow]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "  {:<6} {:>8} {:>8} {:>10}   {}\n",
        "T", "entropy", "unique", "samples", "preview"
    ));
    s.push_str(&format!("  {}\n", "-".repeat(80)));
    for r in rows {
        let preview: Vec<String> = r
            .sampled_tokens
            .iter()
            .take(12)
            .map(|i| format!("t{i}"))
            .collect();
        s.push_str(&format!(
            "  {:<6.2} {:>8.4} {:>8} {:>10}   {}\n",
            r.temperature,
            r.entropy,
            r.unique_count,
            r.sampled_tokens.len(),
            preview.join(" "),
        ));
    }
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_run_temperature_sweep")?;
    println!("=== Recipe: {} ===\n", ctx.name());

    // --- Section 1: Fixed logit distribution -----------------------------
    let logits = canonical_logits();
    println!("--- Canonical Logits ---");
    for (i, v) in logits.iter().enumerate() {
        println!("  t{i:<2}  logit = {v:>6.2}");
    }
    println!();

    // --- Section 2: Sweep across 5 temperatures ------------------------
    let temperatures = [0.0_f32, 0.3, 0.7, 1.0, 1.5];
    let samples_per_temp = 30;
    let mut rows = Vec::with_capacity(temperatures.len());
    for &t in &temperatures {
        rows.push(sweep_at(&logits, t, samples_per_temp, ctx.rng()));
    }

    // --- Section 3: Render the sweep table -----------------------------
    println!(
        "--- Temperature Sweep ({} samples per T) ---\n",
        samples_per_temp
    );
    print!("{}", render_sweep(&rows));
    println!();

    // --- Section 4: Assertion — T=0.0 produces pure argmax ------------
    if let Some(cold) = rows.first() {
        let unique = cold.unique_count;
        if unique != 1 {
            return Err(CookbookError::invalid_format(format!(
                "T=0.0 must be deterministic (unique=1), got unique={unique}"
            )));
        }
        println!("✓ T=0.0 collapses to argmax (unique token count = 1)");
    }

    // --- Section 5: Assertion — entropy is non-decreasing in T ---------
    let mut prev = f32::NEG_INFINITY;
    for r in &rows {
        if r.entropy + 1e-5 < prev {
            return Err(CookbookError::invalid_format(format!(
                "entropy not monotone at T={}: {} < prev {}",
                r.temperature, r.entropy, prev
            )));
        }
        prev = r.entropy;
    }
    println!("✓ Entropy is monotone non-decreasing across the sweep\n");

    // --- Section 6: Persist sweep JSON --------------------------------
    let summary = serde_json::json!({
        "schema_version": 1,
        "sub": "run",
        "logits": logits,
        "rows": rows.iter().map(|r| serde_json::json!({
            "temperature": r.temperature,
            "entropy": r.entropy,
            "unique": r.unique_count,
            "samples": r.sampled_tokens,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("temperature_sweep.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    println!("Wrote {}", path.display());

    // --- Section 7: Metrics ------------------------------------------
    let highest_entropy = rows.last().map_or(0.0, |r| r.entropy);
    let largest_unique = rows.iter().map(|r| r.unique_count).max().unwrap_or(0);
    ctx.record_float_metric("entropy_at_Tmax", f64::from(highest_entropy));
    ctx.record_metric("max_unique_tokens", largest_unique as i64);
    ctx.record_string_metric("verdict", "SWEEP_OK");

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_softmax_temperature_sum_to_one() {
        let probs = softmax_temperature(&canonical_logits(), 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn test_softmax_zero_temperature_is_onehot() {
        let probs = softmax_temperature(&canonical_logits(), 0.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!((probs[0] - 1.0).abs() < 1e-5, "argmax must be 1.0");
        for &p in probs.iter().skip(1) {
            assert!(p.abs() < 1e-6, "non-argmax must be 0");
        }
    }

    #[test]
    fn test_argmax_index_picks_first_max() {
        assert_eq!(argmax_index(&[0.0, 3.0, 1.0, 2.9]), 1);
    }

    #[test]
    fn test_entropy_uniform_is_ln_n() {
        let n = 4;
        let p = vec![1.0 / n as f32; n];
        let h = entropy(&p);
        let expected = (n as f32).ln();
        assert!((h - expected).abs() < 1e-5, "H={h}, expected={expected}");
    }

    #[test]
    fn test_entropy_one_hot_is_zero() {
        let p = vec![1.0_f32, 0.0, 0.0];
        let h = entropy(&p);
        assert!(h.abs() < 1e-6);
    }

    #[test]
    fn test_sweep_at_zero_is_deterministic() {
        let logits = canonical_logits();
        let mut rng = test_rng();
        let row = sweep_at(&logits, 0.0, 20, &mut rng);
        assert_eq!(row.unique_count, 1, "T=0.0 must be deterministic");
        assert!(row.sampled_tokens.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_sweep_entropy_non_decreasing() {
        let logits = canonical_logits();
        let mut rng = test_rng();
        let ts = [0.0_f32, 0.3, 0.7, 1.0, 1.5];
        let rows: Vec<_> = ts
            .iter()
            .map(|&t| sweep_at(&logits, t, 20, &mut rng))
            .collect();
        let mut prev = f32::NEG_INFINITY;
        for r in &rows {
            assert!(
                r.entropy + 1e-5 >= prev,
                "entropy not monotone at T={}: {} < {}",
                r.temperature,
                r.entropy,
                prev
            );
            prev = r.entropy;
        }
    }

    #[test]
    fn test_render_sweep_contains_all_temperatures() {
        let rows = vec![
            SweepRow {
                temperature: 0.0,
                entropy: 0.0,
                sampled_tokens: vec![0; 5],
                unique_count: 1,
            },
            SweepRow {
                temperature: 1.5,
                entropy: 2.0,
                sampled_tokens: vec![0, 1, 2, 3, 4],
                unique_count: 5,
            },
        ];
        let table = render_sweep(&rows);
        assert!(table.contains("0.00"));
        assert!(table.contains("1.50"));
    }
}
