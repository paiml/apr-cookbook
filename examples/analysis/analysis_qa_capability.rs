//! # APR QA Capability Check — CLI equivalent: `apr qa_capability model.apr`
//!
//! Gate 0 pre-flight check: validates that hardware supports a model's required
//! operations before loading weights. Prevents wasted time loading 70B models
//! onto hardware that cannot run them.

use apr_cookbook::prelude::*;
use std::collections::HashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Operations required by a model architecture.
#[derive(Debug, Clone)]
struct ArchConstraints {
    model_name: String,
    required_ops: Vec<String>,
}

/// Hardware profile describing supported operations.
#[derive(Debug, Clone)]
struct HardwareProfile {
    name: String,
    supported_ops: HashSet<String>,
}

/// Capability assessment status for a (model, hardware) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapStatus {
    /// All required ops are supported.
    Supported,
    /// Some ops are missing but partial execution is possible.
    Partial,
    /// Too many ops are missing for useful execution.
    Unsupported,
}

impl fmt::Display for CapStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapStatus::Supported => f.write_str("PASS"),
            CapStatus::Partial => f.write_str("PARTIAL"),
            CapStatus::Unsupported => f.write_str("FAIL"),
        }
    }
}

/// Result of checking one model against one hardware profile.
#[derive(Debug, Clone)]
struct CapabilityResult {
    model: String,
    hardware: String,
    missing_ops: Vec<String>,
    status: CapStatus,
}

// ---------------------------------------------------------------------------
// Fallback recommendations
// ---------------------------------------------------------------------------

/// Suggest a fallback for a missing operation.
fn fallback_for_op(op: &str) -> &str {
    match op {
        "flash_attention" => "Use standard scaled dot-product attention (slower, more memory)",
        "cross_attention" => "Use sequential self-attention with projected KV (approximate)",
        "conv2d" => "Decompose into im2col + matmul on CPU",
        "conv1d" => "Use matmul with Toeplitz matrix on CPU",
        "unet_skip" => "Use residual add on CPU (no skip fusion)",
        "rope" => "Precompute rotation matrix and apply via matmul",
        "rmsnorm" => "Implement as element-wise ops: rsqrt(mean(x^2)) * x",
        "layernorm" => "Implement as element-wise ops: (x - mean) / sqrt(var + eps)",
        "groupnorm" => "Implement as per-group layernorm on CPU",
        "silu" => "Compute as x * sigmoid(x) element-wise on CPU",
        "gelu" => "Use tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))",
        "relu" => "Element-wise max(0, x) on CPU",
        "softmax" => "Row-wise exp(x - max) / sum(exp(x - max)) on CPU",
        "sinusoidal_pe" => "Precompute sin/cos position table on CPU",
        "matmul" => "No fallback: matmul is a fundamental requirement",
        _ => "No known fallback",
    }
}

// ---------------------------------------------------------------------------
// Model architecture definitions
// ---------------------------------------------------------------------------

fn define_model_architectures() -> Vec<ArchConstraints> {
    vec![
        ArchConstraints {
            model_name: "LLaMA".to_string(),
            required_ops: vec![
                "matmul",
                "rope",
                "rmsnorm",
                "silu",
                "softmax",
                "flash_attention",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        },
        ArchConstraints {
            model_name: "Whisper".to_string(),
            required_ops: vec![
                "matmul",
                "layernorm",
                "gelu",
                "softmax",
                "conv1d",
                "sinusoidal_pe",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        },
        ArchConstraints {
            model_name: "StableDiffusion".to_string(),
            required_ops: vec![
                "matmul",
                "groupnorm",
                "silu",
                "conv2d",
                "cross_attention",
                "unet_skip",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Hardware profile definitions
// ---------------------------------------------------------------------------

fn define_hardware_profiles() -> Vec<HardwareProfile> {
    vec![
        HardwareProfile {
            name: "Full GPU".to_string(),
            supported_ops: [
                "matmul",
                "rope",
                "rmsnorm",
                "layernorm",
                "groupnorm",
                "silu",
                "gelu",
                "relu",
                "softmax",
                "flash_attention",
                "cross_attention",
                "conv1d",
                "conv2d",
                "sinusoidal_pe",
                "unet_skip",
            ]
            .iter()
            .map(|s| String::from(*s))
            .collect(),
        },
        HardwareProfile {
            name: "Basic GPU".to_string(),
            supported_ops: ["matmul", "softmax", "relu"]
                .iter()
                .map(|s| String::from(*s))
                .collect(),
        },
        HardwareProfile {
            name: "CPU Only".to_string(),
            supported_ops: [
                "matmul",
                "softmax",
                "layernorm",
                "rmsnorm",
                "relu",
                "gelu",
                "silu",
                "conv1d",
            ]
            .iter()
            .map(|s| String::from(*s))
            .collect(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Capability check logic
// ---------------------------------------------------------------------------

/// Determine the capability status based on the fraction of missing ops.
fn classify_status(total_ops: usize, missing_count: usize) -> CapStatus {
    if missing_count == 0 {
        return CapStatus::Supported;
    }
    // If more than half the ops are missing, it is unsupported.
    let missing_ratio = missing_count as f64 / total_ops.max(1) as f64;
    if missing_ratio > 0.5 {
        CapStatus::Unsupported
    } else {
        CapStatus::Partial
    }
}

/// Check a single model against a single hardware profile.
fn check_capability(arch: &ArchConstraints, hw: &HardwareProfile) -> CapabilityResult {
    let missing_ops: Vec<String> = arch
        .required_ops
        .iter()
        .filter(|op| !hw.supported_ops.contains(op.as_str()))
        .cloned()
        .collect();

    let status = classify_status(arch.required_ops.len(), missing_ops.len());

    CapabilityResult {
        model: arch.model_name.clone(),
        hardware: hw.name.clone(),
        missing_ops,
        status,
    }
}

/// Check all (model, hardware) pairs.
fn check_all_capabilities(
    models: &[ArchConstraints],
    profiles: &[HardwareProfile],
) -> Vec<CapabilityResult> {
    let mut results = Vec::with_capacity(models.len() * profiles.len());
    for model in models {
        for hw in profiles {
            results.push(check_capability(model, hw));
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn print_capability_matrix(
    models: &[ArchConstraints],
    profiles: &[HardwareProfile],
    results: &[CapabilityResult],
) {
    // Header row
    print!("{:<20}", "Model \\ Hardware");
    for hw in profiles {
        print!("{:>14}", hw.name);
    }
    println!();
    println!("{}", "-".repeat(20 + 14 * profiles.len()));

    // Data rows
    for model in models {
        print!("{:<20}", model.model_name);
        for hw in profiles {
            let result = results
                .iter()
                .find(|r| r.model == model.model_name && r.hardware == hw.name);
            let status_str = result.map_or("???", |r| match r.status {
                CapStatus::Supported => "PASS",
                CapStatus::Partial => "PARTIAL",
                CapStatus::Unsupported => "FAIL",
            });
            print!("{:>14}", status_str);
        }
        println!();
    }
}

fn print_failure_details(results: &[CapabilityResult]) {
    let failures: Vec<_> = results
        .iter()
        .filter(|r| r.status != CapStatus::Supported)
        .collect();

    if failures.is_empty() {
        println!("  All combinations are fully supported.");
        return;
    }

    for r in &failures {
        println!(
            "  {} on {}: {} ({} missing op(s))",
            r.model,
            r.hardware,
            r.status,
            r.missing_ops.len(),
        );
        for op in &r.missing_ops {
            println!("    - {}: {}", op, fallback_for_op(op));
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_qa_capability")?;
    println!("=== APR QA Capability Check ===\n");

    // --- Section 1: Define model architectures ---
    println!("--- Model Architectures ---\n");
    let models = define_model_architectures();
    for arch in &models {
        println!(
            "  {}: {} ops [{}]",
            arch.model_name,
            arch.required_ops.len(),
            arch.required_ops.join(", "),
        );
    }

    // --- Section 2: Define hardware profiles ---
    println!("\n--- Hardware Profiles ---\n");
    let profiles = define_hardware_profiles();
    for hw in &profiles {
        let mut sorted_ops: Vec<_> = hw.supported_ops.iter().cloned().collect();
        sorted_ops.sort();
        println!(
            "  {}: {} ops [{}]",
            hw.name,
            hw.supported_ops.len(),
            sorted_ops.join(", "),
        );
    }

    // --- Section 3: Capability matrix ---
    println!("\n--- Capability Matrix ---\n");
    let results = check_all_capabilities(&models, &profiles);
    print_capability_matrix(&models, &profiles, &results);

    // --- Section 4: Failure details and fallback recommendations ---
    println!("\n--- Failure Details & Fallback Recommendations ---\n");
    print_failure_details(&results);

    // --- Section 5: Summary statistics ---
    println!("\n--- Summary ---");
    let total = results.len();
    let pass_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Supported)
        .count();
    let partial_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Partial)
        .count();
    let fail_count = results
        .iter()
        .filter(|r| r.status == CapStatus::Unsupported)
        .count();
    println!("  Total checks:  {total}");
    println!("  Supported:     {pass_count}");
    println!("  Partial:       {partial_count}");
    println!("  Unsupported:   {fail_count}");

    println!("\nCapability check complete.");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "LLaMA")
            .expect("LLaMA architecture must exist")
    }

    fn whisper_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "Whisper")
            .expect("Whisper architecture must exist")
    }

    fn sd_arch() -> ArchConstraints {
        define_model_architectures()
            .into_iter()
            .find(|a| a.model_name == "StableDiffusion")
            .expect("StableDiffusion architecture must exist")
    }

    fn full_gpu() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "Full GPU")
            .expect("Full GPU profile must exist")
    }

    fn basic_gpu() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "Basic GPU")
            .expect("Basic GPU profile must exist")
    }

    fn cpu_only() -> HardwareProfile {
        define_hardware_profiles()
            .into_iter()
            .find(|h| h.name == "CPU Only")
            .expect("CPU Only profile must exist")
    }

    // Test 1: Full GPU supports all models
    #[test]
    fn test_full_gpu_supports_all_models() {
        let hw = full_gpu();
        for arch in &define_model_architectures() {
            let result = check_capability(arch, &hw);
            assert_eq!(
                result.status,
                CapStatus::Supported,
                "{} should be fully supported on Full GPU, missing: {:?}",
                arch.model_name,
                result.missing_ops,
            );
        }
    }

    // Test 2: Basic GPU fails on complex models
    #[test]
    fn test_basic_gpu_fails_stable_diffusion() {
        let result = check_capability(&sd_arch(), &basic_gpu());
        assert_eq!(
            result.status,
            CapStatus::Unsupported,
            "StableDiffusion needs too many ops for Basic GPU",
        );
        assert!(result.missing_ops.len() > 3);
    }

    // Test 3: CPU Only partially supports LLaMA (has rmsnorm, silu but not flash_attention, rope)
    #[test]
    fn test_cpu_only_partial_llama() {
        let result = check_capability(&llama_arch(), &cpu_only());
        assert_eq!(
            result.status,
            CapStatus::Partial,
            "CPU Only should partially support LLaMA",
        );
        assert!(
            result.missing_ops.contains(&"flash_attention".to_string()),
            "flash_attention should be missing on CPU Only",
        );
        assert!(
            result.missing_ops.contains(&"rope".to_string()),
            "rope should be missing on CPU Only",
        );
    }

    // Test 4: Classify status thresholds
    #[test]
    fn test_classify_status_thresholds() {
        assert_eq!(classify_status(6, 0), CapStatus::Supported);
        assert_eq!(classify_status(6, 1), CapStatus::Partial);
        assert_eq!(classify_status(6, 3), CapStatus::Partial);
        assert_eq!(classify_status(6, 4), CapStatus::Unsupported);
        assert_eq!(classify_status(6, 6), CapStatus::Unsupported);
    }

    // Test 5: Empty ops edge case
    #[test]
    fn test_classify_status_zero_ops() {
        // A model with zero required ops is trivially supported
        assert_eq!(classify_status(0, 0), CapStatus::Supported);
    }

    // Test 6: check_all_capabilities returns correct count
    #[test]
    fn test_check_all_count() {
        let models = define_model_architectures();
        let profiles = define_hardware_profiles();
        let results = check_all_capabilities(&models, &profiles);
        assert_eq!(
            results.len(),
            models.len() * profiles.len(),
            "Should have one result per (model, hardware) pair",
        );
    }

    // Test 7: Whisper on CPU Only is partial (missing sinusoidal_pe)
    #[test]
    fn test_whisper_cpu_only_partial() {
        let result = check_capability(&whisper_arch(), &cpu_only());
        assert_eq!(result.status, CapStatus::Partial);
        assert!(
            result.missing_ops.contains(&"sinusoidal_pe".to_string()),
            "sinusoidal_pe should be missing on CPU Only",
        );
    }

    // Test 8: Fallback recommendations exist for all missing ops
    #[test]
    fn test_fallback_recommendations_non_empty() {
        let models = define_model_architectures();
        let profiles = define_hardware_profiles();
        let results = check_all_capabilities(&models, &profiles);
        for r in &results {
            for op in &r.missing_ops {
                let fallback = fallback_for_op(op);
                assert!(
                    !fallback.is_empty(),
                    "Fallback for '{op}' should not be empty",
                );
            }
        }
    }

    // Test 9: CapStatus display formatting
    #[test]
    fn test_cap_status_display() {
        assert_eq!(format!("{}", CapStatus::Supported), "PASS");
        assert_eq!(format!("{}", CapStatus::Partial), "PARTIAL");
        assert_eq!(format!("{}", CapStatus::Unsupported), "FAIL");
    }

    // Test 10: Supported result has empty missing_ops
    #[test]
    fn test_supported_result_has_no_missing_ops() {
        let models = define_model_architectures();
        let hw = full_gpu();
        for arch in &models {
            let result = check_capability(arch, &hw);
            assert!(
                result.missing_ops.is_empty(),
                "{} on Full GPU should have no missing ops, got: {:?}",
                arch.model_name,
                result.missing_ops,
            );
        }
    }
}
