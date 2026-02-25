//! # APR Model QA Gates
//!
//! CLI equivalent: `apr qa model.apr`
//!
//! Runs 6 falsifiable quality gates on an APR model. Each gate has a specific
//! metric and threshold, producing a clear pass/fail result. Designed for
//! CI/CD pipelines where models must meet quality bars before deployment.

use apr_cookbook::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

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
        match self {
            Gate::Format => write!(f, "Format"),
            Gate::Integrity => write!(f, "Integrity"),
            Gate::Performance => write!(f, "Performance"),
            Gate::Size => write!(f, "Size"),
            Gate::Accuracy => write!(f, "Accuracy"),
            Gate::Security => write!(f, "Security"),
        }
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
            max_size_bytes: 100 * 1024 * 1024, // 100 MB
            min_accuracy: 0.80,
        }
    }
}

// ---------------------------------------------------------------------------
// QA gate implementations
// ---------------------------------------------------------------------------

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
    let metric = if valid { 1.0 } else { 0.0 };
    GateResult::new(
        Gate::Format,
        valid,
        metric,
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
        let offset = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        offset.min(bytes.len())
    } else {
        64.min(bytes.len())
    }
}

/// Gate 2: Integrity check (no NaN or Inf values)
fn gate_integrity(bytes: &[u8]) -> GateResult {
    let payload_start = get_payload_offset(bytes);
    let payload = &bytes[payload_start..];

    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    let mut total_floats = 0u64;

    for chunk in payload.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let f = f32::from_bits(bits);
        total_floats += 1;
        if f.is_nan() {
            nan_count += 1;
        }
        if f.is_infinite() {
            inf_count += 1;
        }
    }

    let bad_count = nan_count + inf_count;
    let integrity_ratio = if total_floats > 0 {
        1.0 - (bad_count as f64 / total_floats as f64)
    } else {
        1.0
    };
    let passed = bad_count == 0;

    GateResult::new(
        Gate::Integrity,
        passed,
        integrity_ratio,
        1.0,
        &format!("{total_floats} floats checked: {nan_count} NaN, {inf_count} Inf"),
    )
}

/// Gate 3: Performance check (simulated inference under time budget)
fn gate_performance(bytes: &[u8], max_ms: f64) -> GateResult {
    let payload_start = get_payload_offset(bytes);
    let payload = &bytes[payload_start..];
    let weights: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let n = weights.len();
    let dim = (n as f64).sqrt() as usize;
    let dim = dim.max(1);

    // Generate test input
    let seed = hash_name_to_seed("qa-perf-input");
    let input_bytes = generate_model_payload(seed, dim);
    let input: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Warmup
    let rows = dim.min(weights.len() / dim.max(1));
    let cols = dim.min(input.len());
    let mut output = vec![0.0_f32; dim];
    for r in 0..rows {
        let row_start = r * dim;
        let w_slice = &weights[row_start..weights.len().min(row_start + cols)];
        let sum: f32 = w_slice.iter().zip(&input[..cols]).map(|(w, i)| w * i).sum();
        if r < output.len() {
            output[r] = sum;
        }
    }

    // Timed run
    let iterations: i32 = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        for r in 0..rows {
            let row_start = r * dim;
            let w_slice = &weights[row_start..weights.len().min(row_start + cols)];
            let sum: f32 = w_slice.iter().zip(&input[..cols]).map(|(w, i)| w * i).sum();
            if r < output.len() {
                output[r] = sum;
            }
        }
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iterations);
    let passed = elapsed_ms < max_ms;

    GateResult::new(
        Gate::Performance,
        passed,
        elapsed_ms,
        max_ms,
        &format!("Inference: {elapsed_ms:.3} ms (budget: {max_ms:.1} ms)"),
    )
}

/// Gate 4: Size check (model within size budget)
fn gate_size(bytes: &[u8], max_bytes: usize) -> GateResult {
    let size = bytes.len();
    let passed = size <= max_bytes;
    let size_mb = size as f64 / 1_048_576.0;
    let budget_mb = max_bytes as f64 / 1_048_576.0;

    GateResult::new(
        Gate::Size,
        passed,
        size_mb,
        budget_mb,
        &format!("{size_mb:.2} MB (budget: {budget_mb:.2} MB)"),
    )
}

/// Gate 5: Accuracy check (simulated evaluation against synthetic dataset)
fn gate_accuracy(bytes: &[u8], min_accuracy: f64) -> GateResult {
    let payload_start = get_payload_offset(bytes);
    let payload = &bytes[payload_start..];
    let weights: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let n = weights.len();
    let dim = (n as f64).sqrt() as usize;
    let dim = dim.max(1);

    // Synthetic evaluation: generate test samples and measure classification accuracy
    let num_samples: i32 = 100;
    let mut correct: i32 = 0;
    let acc_rows = dim.min(weights.len() / dim.max(1));
    let acc_cols = dim.min(n);
    for i in 0..num_samples {
        let seed = hash_name_to_seed(&format!("qa-acc-{i}"));
        let input_bytes = generate_model_payload(seed, dim);
        let input: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Forward pass: argmax of matmul output
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0usize;
        for r in 0..acc_rows {
            let row_start = r * dim;
            let w_slice = &weights[row_start..weights.len().min(row_start + acc_cols)];
            let input_slice = &input[..acc_cols.min(input.len())];
            let sum: f32 = w_slice.iter().zip(input_slice).map(|(w, iv)| w * iv).sum();
            if sum > max_val {
                max_val = sum;
                max_idx = r;
            }
        }

        // Synthetic label: deterministic based on seed
        let label = (seed as usize) % acc_rows.max(1);
        if max_idx == label {
            correct += 1;
        }
    }

    let accuracy = f64::from(correct) / f64::from(num_samples);
    // For random weights, accuracy is ~1/dim, which will likely fail the gate
    // unless the threshold is set very low. This demonstrates the gate mechanism.
    let passed = accuracy >= min_accuracy;

    GateResult::new(
        Gate::Accuracy,
        passed,
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

    // Check for embedded executable signatures
    let elf_magic = b"\x7fELF";
    let pe_magic = b"MZ";
    let script_magic = b"#!/";

    if bytes.windows(4).any(|w| w == elf_magic) {
        issues.push("ELF executable signature detected");
    }
    if bytes.len() > 64 && bytes[64..].windows(2).any(|w| w == pe_magic) {
        issues.push("PE executable signature in payload");
    }
    if bytes.windows(3).any(|w| w == script_magic) {
        issues.push("Script shebang detected");
    }

    // Check for suspiciously large zero blocks (potential padding attack)
    let max_zero_run = count_max_zero_run(bytes);
    if max_zero_run > bytes.len() / 2 && bytes.len() > 128 {
        issues.push("Suspiciously large zero block (>50% of file)");
    }

    // Check for embedded URLs/IPs
    if bytes.windows(7).any(|w| w == b"http://") || bytes.windows(8).any(|w| w == b"https://") {
        issues.push("Embedded URL detected in model payload");
    }

    let passed = issues.is_empty();
    let score = if passed { 1.0 } else { 0.0 };
    let detail = if passed {
        "No suspicious patterns detected".to_string()
    } else {
        format!("Issues: {}", issues.join("; "))
    };

    GateResult::new(Gate::Security, passed, score, 1.0, &detail)
}

fn count_max_zero_run(bytes: &[u8]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    for &b in bytes {
        if b == 0 {
            current_run += 1;
            if current_run > max_run {
                max_run = current_run;
            }
        } else {
            current_run = 0;
        }
    }
    max_run
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_qa_gates")?;

    println!("=== APR Model QA Gates ===\n");

    // --- Section 1: Create test model ---
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
            let rec = match gr.gate {
                Gate::Format => "Re-export model with aprender ModelBundleV2",
                Gate::Integrity => "Check training for NaN/Inf; apply gradient clipping",
                Gate::Performance => "Quantize model or reduce layer count",
                Gate::Size => "Apply pruning or knowledge distillation",
                Gate::Accuracy => "Retrain with more data or adjust hyperparameters",
                Gate::Security => "Inspect model provenance; re-export from trusted source",
            };
            println!("  {} (FAIL): {rec}", gr.gate);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_valid_model_passes_format_gate() {
        let bundle = make_valid_bundle();
        let result = gate_format(&bundle);
        assert!(result.passed, "Valid bundle should pass format gate");
    }

    #[test]
    fn test_invalid_magic_fails_format() {
        let mut bundle = make_valid_bundle();
        bundle[0] = b'X';
        let result = gate_format(&bundle);
        assert!(!result.passed);
    }

    #[test]
    fn test_short_file_fails_format() {
        let result = gate_format(&[0x41, 0x50, 0x52, 0x32]); // "APR2" but too short
        assert!(!result.passed);
    }

    #[test]
    fn test_valid_model_passes_integrity() {
        let bundle = make_valid_bundle();
        let result = gate_integrity(&bundle);
        assert!(result.passed, "Clean model should pass integrity");
    }

    #[test]
    fn test_nan_fails_integrity() {
        let mut bundle = make_valid_bundle();
        // Inject NaN at the actual payload region (read payload_offset from header)
        let payload_offset = get_payload_offset(&bundle);
        let nan_bits: u32 = 0x7FC0_0000;
        let nan_bytes = nan_bits.to_le_bytes();
        if payload_offset + 4 <= bundle.len() {
            bundle[payload_offset..payload_offset + 4].copy_from_slice(&nan_bytes);
        }
        let result = gate_integrity(&bundle);
        assert!(!result.passed);
    }

    #[test]
    fn test_performance_gate_passes_with_generous_budget() {
        let bundle = make_valid_bundle();
        let result = gate_performance(&bundle, 10000.0); // 10 seconds
        assert!(result.passed);
    }

    #[test]
    fn test_performance_gate_fails_with_zero_budget() {
        let bundle = make_valid_bundle();
        let result = gate_performance(&bundle, 0.0);
        assert!(!result.passed);
    }

    #[test]
    fn test_size_gate_passes() {
        let bundle = make_valid_bundle();
        let result = gate_size(&bundle, 100 * 1024 * 1024);
        assert!(result.passed);
    }

    #[test]
    fn test_size_gate_fails_tiny_budget() {
        let bundle = make_valid_bundle();
        let result = gate_size(&bundle, 10);
        assert!(!result.passed);
    }

    #[test]
    fn test_accuracy_gate_with_low_threshold() {
        let bundle = make_valid_bundle();
        let result = gate_accuracy(&bundle, 0.0);
        assert!(result.passed, "Zero threshold should always pass");
    }

    #[test]
    fn test_security_gate_clean_model() {
        let bundle = make_valid_bundle();
        let result = gate_security(&bundle);
        assert!(result.passed, "Clean model should pass security gate");
    }

    #[test]
    fn test_security_gate_with_embedded_url() {
        let mut bundle = make_valid_bundle();
        // Inject URL into payload
        let url = b"http://evil.com";
        let offset = 100.min(bundle.len().saturating_sub(url.len()));
        if offset + url.len() <= bundle.len() {
            bundle[offset..offset + url.len()].copy_from_slice(url);
        }
        let result = gate_security(&bundle);
        assert!(!result.passed);
    }

    #[test]
    fn test_all_gates_return_six_results() {
        let bundle = make_valid_bundle();
        let results = run_qa_gates(&bundle);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_gate_result_status_str() {
        let pass = GateResult::new(Gate::Format, true, 1.0, 1.0, "ok");
        let fail = GateResult::new(Gate::Format, false, 0.0, 1.0, "bad");
        assert_eq!(pass.status_str(), "PASS");
        assert_eq!(fail.status_str(), "FAIL");
    }

    #[test]
    fn test_custom_config() {
        let bundle = make_valid_bundle();
        let config = QaConfig {
            max_inference_ms: 50000.0,
            max_size_bytes: 100 * 1024 * 1024,
            min_accuracy: 0.0,
        };
        let results = run_qa_gates_with_config(&bundle, &config);
        let perf = results
            .iter()
            .find(|r| r.gate == Gate::Performance)
            .unwrap();
        assert!(perf.passed, "Generous budget should pass");
    }

    #[test]
    fn test_count_max_zero_run() {
        let data = vec![1, 0, 0, 0, 1, 0, 0, 1];
        assert_eq!(count_max_zero_run(&data), 3);
    }

    #[test]
    fn test_count_max_zero_run_no_zeros() {
        let data = vec![1, 2, 3, 4];
        assert_eq!(count_max_zero_run(&data), 0);
    }

    #[test]
    fn test_gate_display() {
        assert_eq!(format!("{}", Gate::Format), "Format");
        assert_eq!(format!("{}", Gate::Integrity), "Integrity");
        assert_eq!(format!("{}", Gate::Performance), "Performance");
        assert_eq!(format!("{}", Gate::Size), "Size");
        assert_eq!(format!("{}", Gate::Accuracy), "Accuracy");
        assert_eq!(format!("{}", Gate::Security), "Security");
    }
}
