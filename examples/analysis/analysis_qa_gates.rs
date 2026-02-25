//! # APR Model QA Gates — CLI equivalent: `apr qa model.apr`
//!
//! Runs 6 falsifiable quality gates on an APR model for CI/CD pipelines.

use apr_cookbook::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Gate {
    Format,
    Integrity,
    Performance,
    Size,
    Accuracy,
    Security,
}

impl std::fmt::Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Gate::Format => "Format",
            Gate::Integrity => "Integrity",
            Gate::Performance => "Performance",
            Gate::Size => "Size",
            Gate::Accuracy => "Accuracy",
            Gate::Security => "Security",
        })
    }
}

#[derive(Debug, Clone)]
struct GateResult {
    gate: Gate,
    passed: bool,
    metric: f64,
    threshold: f64,
    detail: String,
}

impl GateResult {
    fn new(gate: Gate, passed: bool, metric: f64, threshold: f64, detail: &str) -> Self {
        Self {
            gate,
            passed,
            metric,
            threshold,
            detail: detail.to_string(),
        }
    }
    fn status_str(&self) -> &str {
        if self.passed {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

#[derive(Debug, Clone)]
struct QaConfig {
    max_inference_ms: f64,
    max_size_bytes: usize,
    min_accuracy: f64,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            max_inference_ms: 100.0,
            max_size_bytes: 100 * 1024 * 1024,
            min_accuracy: 0.80,
        }
    }
}

// -- QA gate implementations --

fn run_qa_gates(model_bytes: &[u8]) -> Vec<GateResult> {
    run_qa_gates_with_config(model_bytes, &QaConfig::default())
}

fn run_qa_gates_with_config(model_bytes: &[u8], config: &QaConfig) -> Vec<GateResult> {
    vec![
        gate_format(model_bytes),
        gate_integrity(model_bytes),
        gate_performance(model_bytes, config.max_inference_ms),
        gate_size(model_bytes, config.max_size_bytes),
        gate_accuracy(model_bytes, config.min_accuracy),
        gate_security(model_bytes),
    ]
}

/// Gate 1: Format validation (APR2 magic bytes and minimum structure)
fn gate_format(bytes: &[u8]) -> GateResult {
    let valid = bytes.len() >= 64 && &bytes[0..4] == b"APR2";
    GateResult::new(
        Gate::Format,
        valid,
        if valid { 1.0 } else { 0.0 },
        1.0,
        if valid {
            "Valid APR2 format with proper header"
        } else {
            "Invalid format: missing APR2 magic or header too short"
        },
    )
}

/// Extract the payload offset from APR v2 header (bytes 16-19, u32 LE)
fn get_payload_offset(bytes: &[u8]) -> usize {
    if bytes.len() >= 20 && &bytes[0..4] == b"APR2" {
        (u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize).min(bytes.len())
    } else {
        64.min(bytes.len())
    }
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Gate 2: Integrity check (no NaN or Inf values)
fn gate_integrity(bytes: &[u8]) -> GateResult {
    let payload = &bytes[get_payload_offset(bytes)..];
    let (mut nan_count, mut inf_count, mut total) = (0u64, 0u64, 0u64);
    for chunk in payload.chunks_exact(4) {
        let f = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        total += 1;
        if f.is_nan() {
            nan_count += 1;
        }
        if f.is_infinite() {
            inf_count += 1;
        }
    }
    let bad = nan_count + inf_count;
    let ratio = if total > 0 {
        1.0 - (bad as f64 / total as f64)
    } else {
        1.0
    };
    GateResult::new(
        Gate::Integrity,
        bad == 0,
        ratio,
        1.0,
        &format!("{total} floats checked: {nan_count} NaN, {inf_count} Inf"),
    )
}

/// Gate 3: Performance check (simulated inference under time budget)
fn gate_performance(bytes: &[u8], max_ms: f64) -> GateResult {
    let weights = bytes_to_f32(&bytes[get_payload_offset(bytes)..]);
    let dim = (weights.len() as f64).sqrt().max(1.0) as usize;
    let input = bytes_to_f32(&generate_model_payload(
        hash_name_to_seed("qa-perf-input"),
        dim,
    ));
    let rows = dim.min(weights.len() / dim.max(1));
    let cols = dim.min(input.len());
    let mut output = vec![0.0_f32; dim];
    let run_once = |out: &mut [f32]| {
        for r in 0..rows {
            let s = r * dim;
            let sum: f32 = weights[s..weights.len().min(s + cols)]
                .iter()
                .zip(&input[..cols])
                .map(|(w, i)| w * i)
                .sum();
            if r < out.len() {
                out[r] = sum;
            }
        }
    };
    run_once(&mut output); // warmup
    let iters: i32 = 10;
    let start = Instant::now();
    for _ in 0..iters {
        run_once(&mut output);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    GateResult::new(
        Gate::Performance,
        elapsed_ms < max_ms,
        elapsed_ms,
        max_ms,
        &format!("Inference: {elapsed_ms:.3} ms (budget: {max_ms:.1} ms)"),
    )
}

/// Gate 4: Size check (model within size budget)
fn gate_size(bytes: &[u8], max_bytes: usize) -> GateResult {
    let (size_mb, budget_mb) = (
        bytes.len() as f64 / 1_048_576.0,
        max_bytes as f64 / 1_048_576.0,
    );
    GateResult::new(
        Gate::Size,
        bytes.len() <= max_bytes,
        size_mb,
        budget_mb,
        &format!("{size_mb:.2} MB (budget: {budget_mb:.2} MB)"),
    )
}

/// Gate 5: Accuracy check (simulated evaluation against synthetic dataset)
fn gate_accuracy(bytes: &[u8], min_accuracy: f64) -> GateResult {
    let weights = bytes_to_f32(&bytes[get_payload_offset(bytes)..]);
    let n = weights.len();
    let dim = (n as f64).sqrt().max(1.0) as usize;
    let num_samples: i32 = 100;
    let mut correct: i32 = 0;
    let rows = dim.min(weights.len() / dim.max(1));
    let cols = dim.min(n);
    for i in 0..num_samples {
        let seed = hash_name_to_seed(&format!("qa-acc-{i}"));
        let input = bytes_to_f32(&generate_model_payload(seed, dim));
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0usize;
        for r in 0..rows {
            let s = r * dim;
            let sum: f32 = weights[s..weights.len().min(s + cols)]
                .iter()
                .zip(&input[..cols.min(input.len())])
                .map(|(w, iv)| w * iv)
                .sum();
            if sum > max_val {
                max_val = sum;
                max_idx = r;
            }
        }
        if max_idx == (seed as usize) % rows.max(1) {
            correct += 1;
        }
    }
    let accuracy = f64::from(correct) / f64::from(num_samples);
    GateResult::new(
        Gate::Accuracy,
        accuracy >= min_accuracy,
        accuracy,
        min_accuracy,
        &format!(
            "{correct}/{num_samples} correct ({:.1}%, threshold: {:.1}%)",
            accuracy * 100.0,
            min_accuracy * 100.0
        ),
    )
}

/// Gate 6: Security check (no suspicious patterns)
fn gate_security(bytes: &[u8]) -> GateResult {
    let mut issues = Vec::new();
    if bytes.windows(4).any(|w| w == b"\x7fELF") {
        issues.push("ELF executable signature detected");
    }
    if bytes.len() > 64 && bytes[64..].windows(2).any(|w| w == b"MZ") {
        issues.push("PE executable signature in payload");
    }
    if bytes.windows(3).any(|w| w == b"#!/") {
        issues.push("Script shebang detected");
    }
    if count_max_zero_run(bytes) > bytes.len() / 2 && bytes.len() > 128 {
        issues.push("Suspiciously large zero block (>50% of file)");
    }
    if bytes.windows(7).any(|w| w == b"http://") || bytes.windows(8).any(|w| w == b"https://") {
        issues.push("Embedded URL detected in model payload");
    }
    let passed = issues.is_empty();
    let detail = if passed {
        "No suspicious patterns detected".into()
    } else {
        format!("Issues: {}", issues.join("; "))
    };
    GateResult::new(
        Gate::Security,
        passed,
        if passed { 1.0 } else { 0.0 },
        1.0,
        &detail,
    )
}

fn count_max_zero_run(bytes: &[u8]) -> usize {
    let (mut max_run, mut cur) = (0, 0);
    for &b in bytes {
        if b == 0 {
            cur += 1;
            max_run = max_run.max(cur);
        } else {
            cur = 0;
        }
    }
    max_run
}

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
