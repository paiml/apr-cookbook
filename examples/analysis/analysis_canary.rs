//! # Canary Tokens for Regression Testing
//! **CLI Equivalent**: `apr canary`
//!
//! Embeds test vectors in a model and verifies outputs match expected values,
//! detecting model drift and weight corruption.
//!
//! ## CLI equivalent
//! ```bash
//! apr canary create model.apr
//! apr canary check model.apr
//! ```
//!
//! ## What this demonstrates
//! - Deterministic canary generation from model weights
//! - Tolerance-based drift detection
//! - JSON serialization of canary test vectors
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Canary {
    input: Vec<f32>,
    expected_output: Vec<f32>,
    tolerance: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CanaryReport {
    total: usize,
    passed: usize,
    failed: usize,
    results: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Canary creation
// ---------------------------------------------------------------------------

/// Create canary test vectors from model weights.
///
/// Each canary uses a deterministic slice of the weights as input,
/// and computes an expected output via a simple dot-product probe.
fn create_canaries(model_weights: &[f32], n: usize, tolerance: f32) -> Vec<Canary> {
    if model_weights.is_empty() {
        return vec![];
    }

    let mut canaries = Vec::with_capacity(n);
    let chunk_size = (model_weights.len() / n.max(1)).max(1);

    for i in 0..n {
        let start = (i * chunk_size) % model_weights.len();
        let end = (start + chunk_size.min(8)).min(model_weights.len());
        let input: Vec<f32> = model_weights[start..end].to_vec();

        // Compute expected output: weighted sum probe
        let expected = compute_probe(&input, model_weights);

        canaries.push(Canary {
            input,
            expected_output: expected,
            tolerance,
        });
    }

    canaries
}

/// Simple probe function: dot product of input with cyclic weight slice.
fn compute_probe(input: &[f32], weights: &[f32]) -> Vec<f32> {
    let sum: f32 = input
        .iter()
        .enumerate()
        .map(|(i, &v)| v * weights[i % weights.len()])
        .sum();

    // Return a single-element output normalized by input length
    let len = input.len().max(1) as f32;
    vec![sum / len]
}

/// Check canaries against current model weights.
fn check_canaries(model_weights: &[f32], canaries: &[Canary]) -> Vec<bool> {
    canaries
        .iter()
        .map(|canary| {
            let actual = compute_probe(&canary.input, model_weights);
            if actual.len() != canary.expected_output.len() {
                return false;
            }
            actual
                .iter()
                .zip(canary.expected_output.iter())
                .all(|(a, e)| (a - e).abs() <= canary.tolerance)
        })
        .collect()
}

/// Serialize canaries to JSON string.
fn canaries_to_json(canaries: &[Canary]) -> String {
    let entries: Vec<String> = canaries
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "  {{\n    \"id\": {},\n    \"input\": {:?},\n    \"expected\": {:?},\n    \"tolerance\": {}\n  }}",
                i, c.input, c.expected_output, c.tolerance
            )
        })
        .collect();
    format!("[\n{}\n]", entries.join(",\n"))
}

/// Deserialize canaries from JSON string (simple parser).
fn canaries_from_json(json: &str) -> Vec<Canary> {
    let mut canaries = Vec::new();
    let mut input = Vec::new();
    let mut expected = Vec::new();
    let mut tolerance = 1e-6_f32;

    for line in json.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("\"input\":") {
            let arr = extract_f32_array(trimmed);
            input = arr;
        } else if trimmed.starts_with("\"expected\":") {
            let arr = extract_f32_array(trimmed);
            expected = arr;
        } else if trimmed.starts_with("\"tolerance\":") {
            let val = trimmed
                .trim_start_matches("\"tolerance\":")
                .trim()
                .trim_end_matches(',')
                .trim();
            tolerance = val.parse().unwrap_or(1e-6);
        } else if (trimmed == "}" || trimmed == "},") && !input.is_empty() {
            canaries.push(Canary {
                input: input.clone(),
                expected_output: expected.clone(),
                tolerance,
            });
            input.clear();
            expected.clear();
        }
    }

    canaries
}

fn extract_f32_array(s: &str) -> Vec<f32> {
    let start = s.find('[').unwrap_or(0);
    let end = s.find(']').unwrap_or(s.len());
    let inner = &s[start + 1..end];
    inner
        .split(',')
        .filter_map(|v| v.trim().parse::<f32>().ok())
        .collect()
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_canary")?;

    // ── Section 1: Create a synthetic model ─────────────────────────────
    println!("=== Canary Token Regression Testing ===\n");

    let model_weights: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    println!("Model size: {} weights", model_weights.len());

    // ── Section 2: Create canaries ──────────────────────────────────────
    println!("\n--- Canary Creation ---");
    let canaries = create_canaries(&model_weights, 5, 1e-6);
    println!("Created {} canaries", canaries.len());
    for (i, c) in canaries.iter().enumerate() {
        println!(
            "  Canary {}: input_len={}, expected={:?}, tol={}",
            i,
            c.input.len(),
            c.expected_output,
            c.tolerance
        );
    }

    // ── Section 3: Save canaries to JSON ────────────────────────────────
    println!("\n--- Canary JSON Serialization ---");
    let json = canaries_to_json(&canaries);
    println!("JSON ({} bytes):", json.len());
    println!("{}", &json[..json.len().min(300)]);

    // ── Section 4: Verify canaries pass on original model ───────────────
    println!("\n--- Check Original Model ---");
    let results = check_canaries(&model_weights, &canaries);
    let passed = results.iter().filter(|&&r| r).count();
    println!("Results: {}/{} passed", passed, results.len());
    for (i, &r) in results.iter().enumerate() {
        println!("  Canary {}: {}", i, if r { "PASS" } else { "FAIL" });
    }

    // ── Section 5: Modify model and detect drift ────────────────────────
    println!("\n--- Check Modified Model (Drift Detection) ---");
    let mut modified_weights = model_weights.clone();
    // Corrupt several weights
    for w in modified_weights.iter_mut().take(32) {
        *w += 10.0;
    }
    let drift_results = check_canaries(&modified_weights, &canaries);
    let drift_failed = drift_results.iter().filter(|&&r| !r).count();
    println!(
        "After corruption: {}/{} canaries detected drift",
        drift_failed,
        drift_results.len()
    );

    // ── Section 6: JSON roundtrip ───────────────────────────────────────
    println!("\n--- JSON Roundtrip ---");
    let restored = canaries_from_json(&json);
    println!("Restored {} canaries from JSON", restored.len());
    let roundtrip_results = check_canaries(&model_weights, &restored);
    let roundtrip_passed = roundtrip_results.iter().filter(|&&r| r).count();
    println!(
        "Roundtrip check: {}/{} passed",
        roundtrip_passed,
        roundtrip_results.len()
    );

    // Fingerprint
    let mut hasher = DefaultHasher::new();
    canaries.len().hash(&mut hasher);
    passed.hash(&mut hasher);
    drift_failed.hash(&mut hasher);
    println!("\nCanary fingerprint: {:016x}", hasher.finish());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> Vec<f32> {
        (0..256).map(|i| (i as f32 * 0.01).sin()).collect()
    }

    #[test]
    fn test_canaries_pass_on_original_model() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 5, 1e-6);
        let results = check_canaries(&weights, &canaries);
        assert!(
            results.iter().all(|&r| r),
            "all canaries should pass on original model"
        );
    }

    #[test]
    fn test_canaries_fail_on_modified_model() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 5, 1e-6);

        let mut modified = weights.clone();
        for w in modified.iter_mut().take(64) {
            *w += 100.0;
        }

        let results = check_canaries(&modified, &canaries);
        let any_failed = results.iter().any(|&r| !r);
        assert!(
            any_failed,
            "at least one canary should fail after corruption"
        );
    }

    #[test]
    fn test_tolerance_allows_small_drift() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 3, 100.0); // very large tolerance

        let mut modified = weights.clone();
        modified[0] += 0.001;

        let results = check_canaries(&modified, &canaries);
        let all_pass = results.iter().all(|&r| r);
        assert!(all_pass, "large tolerance should accept small drift");
    }

    #[test]
    fn test_deterministic_creation() {
        let weights = sample_weights();
        let c1 = create_canaries(&weights, 4, 1e-6);
        let c2 = create_canaries(&weights, 4, 1e-6);

        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.input, b.input);
            assert_eq!(a.expected_output, b.expected_output);
            assert_eq!(a.tolerance, b.tolerance);
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let weights = sample_weights();
        let original = create_canaries(&weights, 3, 1e-6);
        let json = canaries_to_json(&original);
        let restored = canaries_from_json(&json);

        assert_eq!(original.len(), restored.len());
        for (a, b) in original.iter().zip(restored.iter()) {
            assert_eq!(a.input.len(), b.input.len());
            for (x, y) in a.input.iter().zip(b.input.iter()) {
                assert!((x - y).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_empty_weights() {
        let canaries = create_canaries(&[], 5, 1e-6);
        assert!(canaries.is_empty());
    }

    #[test]
    fn test_single_weight() {
        let weights = vec![42.0_f32];
        let canaries = create_canaries(&weights, 1, 1e-6);
        assert_eq!(canaries.len(), 1);
        let results = check_canaries(&weights, &canaries);
        assert!(results[0]);
    }

    #[test]
    fn test_zero_canaries_requested() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 0, 1e-6);
        assert!(canaries.is_empty());
    }

    #[test]
    fn test_canary_report_counts() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 5, 1e-6);
        let results = check_canaries(&weights, &canaries);

        let report = CanaryReport {
            total: results.len(),
            passed: results.iter().filter(|&&r| r).count(),
            failed: results.iter().filter(|&&r| !r).count(),
            results: results.clone(),
        };

        assert_eq!(report.total, 5);
        assert_eq!(report.passed + report.failed, report.total);
    }

    #[test]
    fn test_json_format_valid() {
        let weights = sample_weights();
        let canaries = create_canaries(&weights, 2, 0.001);
        let json = canaries_to_json(&canaries);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"input\""));
        assert!(json.contains("\"expected\""));
        assert!(json.contains("\"tolerance\""));
    }
}
