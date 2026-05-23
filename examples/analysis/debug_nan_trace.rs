//! # Recipe: Gradient NaN Trace
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr debug model.apr --trace-gradients --detect-nan`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example debug_nan_trace` exits 0
//! 2. [x] `cargo test --example debug_nan_trace` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr debug` gradient tracing in-process (no shell-out)
//! 10. [x] Unit tests cover NaN / Inf / underflow / clean path
//!
//! ## Learning Objective
//! Traces gradient flow through a multi-layer model, detecting NaN, +/-Inf, and
//! underflow (gradients < 1e-30). Reports the first layer where corruption
//! occurs -- a common class of silent training failures.
//!
//! ## Run Command
//! ```bash
//! cargo run --example debug_nan_trace
//! ```
//!
//! ## References
//! - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press, Chapter 8 (Optimization).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Health {
    Clean,
    Underflow,
    Nan,
    Inf,
}

impl Health {
    fn label(self) -> &'static str {
        match self {
            Self::Clean => "CLEAN",
            Self::Underflow => "UNDERFLOW",
            Self::Nan => "NAN",
            Self::Inf => "INF",
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LayerGrad {
    name: String,
    grad: Vec<f32>,
    health: Health,
    max_abs: f32,
    min_abs_nonzero: f32,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn classify_grad(grad: &[f32]) -> Health {
    if grad.iter().any(|v| v.is_nan()) {
        return Health::Nan;
    }
    if grad.iter().any(|v| v.is_infinite()) {
        return Health::Inf;
    }
    let min_nonzero = grad
        .iter()
        .filter(|v| v.abs() > 0.0)
        .copied()
        .map(f32::abs)
        .fold(f32::INFINITY, f32::min);
    if min_nonzero < 1e-30 && min_nonzero > 0.0 {
        return Health::Underflow;
    }
    Health::Clean
}

fn analyze_layer(name: &str, grad: Vec<f32>) -> LayerGrad {
    let health = classify_grad(&grad);
    let max_abs = grad.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let min_abs_nonzero = grad
        .iter()
        .filter(|v| v.abs() > 0.0 && !v.is_nan())
        .copied()
        .map(f32::abs)
        .fold(f32::INFINITY, f32::min);
    LayerGrad {
        name: name.into(),
        grad,
        health,
        max_abs,
        min_abs_nonzero,
    }
}

fn find_first_bad_layer(layers: &[LayerGrad]) -> Option<usize> {
    layers.iter().position(|l| l.health != Health::Clean)
}

/// Build a synthetic multi-layer gradient trace with an injected fault.
fn build_trace(inject_fault: bool) -> Vec<LayerGrad> {
    let dim = 16;
    let seed = hash_name_to_seed("debug-nan-trace");
    let bytes = generate_model_payload(seed, dim * 6);
    let mut layers = Vec::new();
    for i in 0..6 {
        let mut grad: Vec<f32> = bytes[i * dim * 4..(i + 1) * dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * 0.01)
            .collect();
        if inject_fault && i == 3 {
            // Inject a NaN into the 3rd layer.
            grad[2] = f32::NAN;
        }
        if inject_fault && i == 4 {
            grad[0] = f32::INFINITY;
        }
        layers.push(analyze_layer(&format!("layer_{i}"), grad));
    }
    layers
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("debug_nan_trace")?;
    println!("=== Recipe: {} ===", ctx.name());

    let clean = build_trace(false);
    let faulty = build_trace(true);

    println!("\n--- Clean Trace ---");
    print_trace(&clean);
    println!("\n--- Faulty Trace ---");
    print_trace(&faulty);

    let first_bad = find_first_bad_layer(&faulty);
    match first_bad {
        Some(idx) => println!(
            "\nFirst bad layer: {} ({})",
            faulty[idx].name,
            faulty[idx].health.label()
        ),
        None => println!("\nNo bad layers."),
    }

    // Sanity.
    assert_eq!(find_first_bad_layer(&clean), None);
    assert_eq!(first_bad, Some(3));

    let out = json!({
        "recipe": ctx.name(),
        "clean_layers": clean.len(),
        "faulty_layers": faulty.len(),
        "first_bad_index": first_bad,
        "first_bad_name": first_bad.map(|i| faulty[i].name.clone()),
        "faulty_trace": faulty.iter().map(|l| json!({
            "name": l.name,
            "health": l.health.label(),
            "max_abs": l.max_abs,
            "min_abs_nonzero": if l.min_abs_nonzero.is_finite() { l.min_abs_nonzero } else { 0.0 },
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("nan-trace.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

fn print_trace(layers: &[LayerGrad]) {
    println!(
        "{:>10} {:>10} {:>14} {:>14}",
        "Layer", "Health", "MaxAbs", "MinAbsNonzero"
    );
    for l in layers {
        println!(
            "{:>10} {:>10} {:>14.4e} {:>14.4e}",
            l.name,
            l.health.label(),
            l.max_abs,
            if l.min_abs_nonzero.is_finite() {
                l.min_abs_nonzero
            } else {
                0.0
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_clean() {
        let g = [1.0_f32, 2.0, -3.0, 0.5];
        assert_eq!(classify_grad(&g), Health::Clean);
    }

    #[test]
    fn test_classify_nan() {
        let g = [1.0_f32, f32::NAN, 3.0];
        assert_eq!(classify_grad(&g), Health::Nan);
    }

    #[test]
    fn test_classify_inf() {
        let g = [1.0_f32, f32::INFINITY, 3.0];
        assert_eq!(classify_grad(&g), Health::Inf);
    }

    #[test]
    fn test_classify_underflow() {
        let g = [1.0_f32, 1e-35, 2.0];
        assert_eq!(classify_grad(&g), Health::Underflow);
    }

    #[test]
    fn test_first_bad_layer_finds_nan() {
        let layers = build_trace(true);
        assert_eq!(find_first_bad_layer(&layers), Some(3));
    }

    #[test]
    fn test_clean_trace_no_bad_layer() {
        let layers = build_trace(false);
        assert_eq!(find_first_bad_layer(&layers), None);
    }

    #[test]
    fn test_analyze_layer_records_max_abs() {
        let layer = analyze_layer("l", vec![-0.5_f32, 0.2, 1.5]);
        assert!((layer.max_abs - 1.5).abs() < 1e-6);
    }
}
