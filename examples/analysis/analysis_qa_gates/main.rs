#![allow(unused_imports)]
//! # APR Model QA Gates — CLI equivalent: `apr qa model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Runs 6 falsifiable quality gates on an APR model for CI/CD pipelines.
//!
//!
//! ## Format Variants
//! ```bash
//! apr qa model.apr          # APR native format
//! apr qa model.gguf         # GGUF (llama.cpp compatible)
//! apr qa model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_qa_gates")?;
    println!("=== APR Model QA Gates ===\n");
    let dim: usize = 64;
    let seed = hash_name_to_seed("qa-model");
    let weight_bytes = generate_model_payload(seed, dim * dim);
    let bias_bytes = generate_model_payload(seed + 1, dim);

    let bundle = ModelBundleV2::new()
        .with_name("qa-target")
        .with_description("Model for QA gate testing")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], weight_bytes)
        .add_tensor("bias", vec![dim], bias_bytes)
        .build();

    let model_path = ctx.path("qa-target.apr");
    std::fs::write(&model_path, &bundle)?;
    println!("Model: qa-target ({} bytes)\n", bundle.len());

    // --- Section 2: Run QA gates with default config ---
    println!("--- Gate-by-Gate Results ---\n");
    let results = run_qa_gates(&bundle);

    println!(
        "{:<15} {:<6} {:>10} {:>10} Detail",
        "Gate", "Status", "Metric", "Threshold"
    );
    println!("{}", "-".repeat(80));

    for gr in &results {
        println!(
            "{:<15} {:<6} {:>10.4} {:>10.4} {}",
            gr.gate,
            gr.status_str(),
            gr.metric,
            gr.threshold,
            gr.detail,
        );
    }

    // --- Section 3: Pass/fail summary ---
    println!("\n--- Summary ---");
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;
    println!("Total gates: {total}");
    println!("Passed:      {passed}");
    println!("Failed:      {failed}");
    println!(
        "Overall:     {}",
        if failed == 0 {
            "ALL GATES PASSED"
        } else {
            "GATES FAILED"
        }
    );

    // --- Section 4: Recommendations for failures ---
    println!("\n--- Recommendations ---");
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if failures.is_empty() {
        println!("  No failures. Model is deployment-ready.");
    } else {
        for gr in &failures {
            println!("  {} (FAIL): needs attention", gr.gate);
        }
    }

    // --- Section 5: Run with custom thresholds ---
    println!("\n--- Custom Threshold Run ---");
    let strict_config = QaConfig {
        max_inference_ms: 1.0, // very strict
        max_size_bytes: 1024,  // very small
        min_accuracy: 0.01,    // lenient (random model)
    };
    let strict_results = run_qa_gates_with_config(&bundle, &strict_config);
    for gr in &strict_results {
        println!(
            "  {}: {} (metric={:.4}, threshold={:.4})",
            gr.gate,
            gr.status_str(),
            gr.metric,
            gr.threshold
        );
    }

    println!("\nQA gates complete.");
    ctx.report()?;
    Ok(())
}

// -- Tests --

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_bundle() -> Vec<u8> {
        let seed = hash_name_to_seed("qa-test");
        let payload = generate_model_payload(seed, 32 * 32);
        ModelBundleV2::new()
            .with_name("qa-test")
            .with_description("test model for QA")
            .with_compression(Compression::None)
            .with_quantization(Quantization::FP32)
            .add_tensor("weight", vec![32, 32], payload)
            .build()
    }

    #[test]
    fn test_format_gate_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert!(gate_format(&bundle).passed, "Valid bundle should pass");
        // Invalid magic
        let mut bad = bundle.clone();
        bad[0] = b'X';
        assert!(!gate_format(&bad).passed);
        // Too short
        assert!(!gate_format(&[0x41, 0x50, 0x52, 0x32]).passed);
    }

    #[test]
    fn test_integrity_gate_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert!(gate_integrity(&bundle).passed, "Clean model should pass");
        // Inject NaN
        let mut bad = bundle;
        let offset = get_payload_offset(&bad);
        let nan_bytes = 0x7FC0_0000_u32.to_le_bytes();
        if offset + 4 <= bad.len() {
            bad[offset..offset + 4].copy_from_slice(&nan_bytes);
        }
        assert!(!gate_integrity(&bad).passed);
    }

    #[test]
    fn test_performance_gate_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert!(gate_performance(&bundle, 10000.0).passed);
        assert!(!gate_performance(&bundle, 0.0).passed);
    }

    #[test]
    fn test_size_gate_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert!(gate_size(&bundle, 100 * 1024 * 1024).passed);
        assert!(!gate_size(&bundle, 10).passed);
    }

    #[test]
    fn test_accuracy_gate_with_low_threshold() {
        let bundle = make_valid_bundle();
        assert!(
            gate_accuracy(&bundle, 0.0).passed,
            "Zero threshold should always pass"
        );
    }

    #[test]
    fn test_security_gate_pass_and_fail() {
        let bundle = make_valid_bundle();
        assert!(gate_security(&bundle).passed, "Clean model should pass");
        // Inject URL
        let mut bad = bundle;
        let url = b"http://evil.com";
        let off = 100.min(bad.len().saturating_sub(url.len()));
        if off + url.len() <= bad.len() {
            bad[off..off + url.len()].copy_from_slice(url);
        }
        assert!(!gate_security(&bad).passed);
    }

    #[test]
    fn test_run_qa_gates_returns_six_with_custom_config() {
        let bundle = make_valid_bundle();
        assert_eq!(run_qa_gates(&bundle).len(), 6);
        let config = QaConfig {
            max_inference_ms: 50000.0,
            max_size_bytes: 100 * 1024 * 1024,
            min_accuracy: 0.0,
        };
        let results = run_qa_gates_with_config(&bundle, &config);
        assert_eq!(results.len(), 6);
        let perf = results
            .iter()
            .find(|r| r.gate == Gate::Performance)
            .unwrap();
        assert!(perf.passed, "Generous budget should pass");
    }

    #[test]
    fn test_count_max_zero_run_and_status_str() {
        assert_eq!(count_max_zero_run(&[1, 0, 0, 0, 1, 0, 0, 1]), 3);
        assert_eq!(count_max_zero_run(&[1, 2, 3, 4]), 0);
        let pass = GateResult::new(Gate::Format, true, 1.0, 1.0, "ok");
        let fail = GateResult::new(Gate::Format, false, 0.0, 1.0, "bad");
        assert_eq!(pass.status_str(), "PASS");
        assert_eq!(fail.status_str(), "FAIL");
    }
}
