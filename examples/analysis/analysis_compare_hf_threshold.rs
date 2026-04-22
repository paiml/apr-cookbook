//! # Recipe: compare-hf Threshold Sweep
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr compare-hf model.apr --repo my-org/my-model --threshold <tau>`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example analysis_compare_hf_threshold` exits 0
//! 2. [x] `cargo test --example analysis_compare_hf_threshold` passes
//! 3. [x] Deterministic output (same seed → same mismatch counts)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Threshold sweep is monotone (stricter τ → ≥ mismatches)
//!
//! ## Learning Objective
//! Demonstrates how the `--threshold` knob on `apr compare-hf` trades off
//! false-positive mismatches versus silent drift. We run the full APR-vs-HF
//! comparison four times at τ ∈ {1e-3, 1e-4, 1e-5, 1e-6} with fixed injected
//! noise, then chart the mismatched-tensor count. The sweep is monotone
//! non-decreasing as τ tightens — a foundational property engineers rely on
//! when picking a release-blocking threshold in CI.
//!
//! ## Run Command
//! ```bash
//! cargo run --example analysis_compare_hf_threshold
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr compare-hf model.apr          --threshold 1e-5   # APR ↔ HF SafeTensors (tight)
//! apr compare-hf model.apr          --threshold 1e-3   # APR ↔ HF SafeTensors (loose)
//! apr compare-hf model.gguf         --threshold 1e-5   # GGUF ↔ HF
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;

/// Per-tensor comparison outcome at a fixed threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorOutcome {
    pub name: String,
    pub max_abs_err: f64,
    pub mismatched: bool,
}

/// Sweep result at a single threshold τ.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepPoint {
    pub threshold: f64,
    pub total: usize,
    pub mismatched: usize,
}

/// Build `n` synthetic tensor error magnitudes in a deterministic spread.
///
/// Errors range from near-zero (clean tensors) to larger deliberate drift in
/// a handful — simulating a real APR vs HF comparison where most tensors
/// match bit-for-bit but a few have accumulated float error.
pub fn synthesize_errors(rng: &mut impl Rng, n: usize) -> Vec<TensorOutcome> {
    (0..n)
        .map(|i| {
            // Deterministic magnitude spread: most tiny, a few medium, one or two large.
            let mag_exp: f64 = match i {
                0..=2 => -8.0 + rng.gen::<f64>() * 1.0, // 1e-8..1e-7 (clean)
                3..=5 => -6.0 + rng.gen::<f64>() * 1.0, // 1e-6..1e-5 (borderline)
                6..=7 => -4.5 + rng.gen::<f64>() * 0.5, // 3e-5..1e-4 (medium)
                _ => -3.5 + rng.gen::<f64>() * 0.5,     // 3e-4..1e-3 (large)
            };
            let mag = 10.0_f64.powf(mag_exp);
            TensorOutcome {
                name: format!("tensor_{i:02}"),
                max_abs_err: mag,
                mismatched: false,
            }
        })
        .collect()
}

/// Apply a threshold τ — mark every tensor with `max_abs_err > τ` as mismatched.
#[must_use]
pub fn apply_threshold(tensors: &[TensorOutcome], threshold: f64) -> SweepPoint {
    let mut mismatched = 0;
    for t in tensors {
        if t.max_abs_err > threshold {
            mismatched += 1;
        }
    }
    SweepPoint {
        threshold,
        total: tensors.len(),
        mismatched,
    }
}

/// Run the full sweep across a list of thresholds.
#[must_use]
pub fn run_sweep(tensors: &[TensorOutcome], thresholds: &[f64]) -> Vec<SweepPoint> {
    thresholds
        .iter()
        .map(|&t| apply_threshold(tensors, t))
        .collect()
}

/// Render a simple bar chart of mismatched-tensor counts per threshold.
#[must_use]
pub fn render_chart(sweep: &[SweepPoint]) -> String {
    let mut s = String::new();
    s.push_str("  τ          mismatched  chart\n");
    s.push_str("  ---------  ----------  ----------------\n");
    let max_count = sweep.iter().map(|p| p.mismatched).max().unwrap_or(0).max(1);
    for p in sweep {
        let bar_len = p.mismatched * 20 / max_count;
        let bar: String = "#".repeat(bar_len);
        s.push_str(&format!(
            "  {:<9.0e}  {:>10}  {}\n",
            p.threshold, p.mismatched, bar
        ));
    }
    s
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_compare_hf_threshold")?;
    println!("=== Recipe: {} ===\n", ctx.name());

    // --- Section 1: Synthesize tensor-error spread -------------------------
    let n_tensors = 10;
    let tensors = synthesize_errors(ctx.rng(), n_tensors);
    println!("Synthesized {n_tensors} tensor error magnitudes:");
    for t in &tensors {
        println!("  {:<12} max_abs_err = {:.2e}", t.name, t.max_abs_err);
    }
    println!();

    // --- Section 2: Sweep four canonical thresholds ------------------------
    let thresholds = [1e-3, 1e-4, 1e-5, 1e-6];
    let sweep = run_sweep(&tensors, &thresholds);

    println!("--- Threshold Sweep ---");
    println!("{}", render_chart(&sweep));

    // --- Section 3: Assert monotone non-decreasing as τ tightens ----------
    let mut prev = 0usize;
    for p in &sweep {
        if p.mismatched < prev {
            return Err(CookbookError::invalid_format(format!(
                "sweep not monotone at τ={:.0e}: {} < prev {}",
                p.threshold, p.mismatched, prev
            )));
        }
        prev = p.mismatched;
    }
    println!("✓ Sweep is monotone: stricter τ → more (or equal) mismatches\n");

    // --- Section 4: Persist sweep JSON for CI dashboards ------------------
    let summary = serde_json::json!({
        "schema_version": 1,
        "sub": "compare-hf",
        "total_tensors": n_tensors,
        "points": sweep.iter().map(|p| serde_json::json!({
            "threshold": p.threshold,
            "mismatched": p.mismatched,
            "total": p.total,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("compare_hf_sweep.json");
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    println!("Sweep JSON: {}\n", out_path.display());

    // --- Section 5: Metrics ------------------------------------------------
    let tightest = sweep.last().map_or(0, |p| p.mismatched);
    let loosest = sweep.first().map_or(0, |p| p.mismatched);
    ctx.record_metric("tensors_total", n_tensors as i64);
    ctx.record_metric("mismatched_tightest", tightest as i64);
    ctx.record_metric("mismatched_loosest", loosest as i64);
    ctx.record_string_metric(
        "verdict",
        if tightest >= loosest {
            "MONOTONE"
        } else {
            "BROKEN"
        },
    );

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
    fn test_synthesize_errors_produces_n() {
        let mut rng = test_rng();
        let xs = synthesize_errors(&mut rng, 10);
        assert_eq!(xs.len(), 10);
        for x in &xs {
            assert!(x.max_abs_err > 0.0);
            assert!(x.max_abs_err.is_finite());
        }
    }

    #[test]
    fn test_apply_threshold_counts_correctly() {
        let tensors = vec![
            TensorOutcome {
                name: "a".into(),
                max_abs_err: 1e-2,
                mismatched: false,
            },
            TensorOutcome {
                name: "b".into(),
                max_abs_err: 1e-6,
                mismatched: false,
            },
            TensorOutcome {
                name: "c".into(),
                max_abs_err: 1e-8,
                mismatched: false,
            },
        ];
        let p = apply_threshold(&tensors, 1e-5);
        // 1e-2 > 1e-5 → mismatched; 1e-6, 1e-8 ≤ 1e-5 → match.
        assert_eq!(p.mismatched, 1);
        assert_eq!(p.total, 3);
    }

    #[test]
    fn test_sweep_is_monotone_non_decreasing() {
        let mut rng = test_rng();
        let tensors = synthesize_errors(&mut rng, 10);
        let thresholds = [1e-3, 1e-4, 1e-5, 1e-6];
        let sweep = run_sweep(&tensors, &thresholds);
        let mut prev = 0usize;
        for p in &sweep {
            assert!(
                p.mismatched >= prev,
                "not monotone at τ={:.0e}: {} < {}",
                p.threshold,
                p.mismatched,
                prev
            );
            prev = p.mismatched;
        }
    }

    #[test]
    fn test_render_chart_includes_header_and_rows() {
        let sweep = vec![
            SweepPoint {
                threshold: 1e-3,
                total: 10,
                mismatched: 2,
            },
            SweepPoint {
                threshold: 1e-5,
                total: 10,
                mismatched: 6,
            },
        ];
        let chart = render_chart(&sweep);
        assert!(chart.contains("mismatched"));
        assert!(chart.contains("1e-3"));
        assert!(chart.contains("1e-5"));
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let xs1 = synthesize_errors(&mut test_rng(), 10);
        let xs2 = synthesize_errors(&mut test_rng(), 10);
        assert_eq!(xs1, xs2);
    }
}
