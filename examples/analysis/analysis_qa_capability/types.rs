//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::collections::HashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Operations required by a model architecture.
#[derive(Debug, Clone)]
pub struct ArchConstraints {
    pub model_name: String,
    pub required_ops: Vec<String>,
}

/// Hardware profile describing supported operations.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub name: String,
    pub supported_ops: HashSet<String>,
}

/// Capability assessment status for a (model, hardware) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapStatus {
    // All required ops are supported.
    Supported,
    // Some ops are missing but partial execution is possible.
    Partial,
    // Too many ops are missing for useful execution.
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
pub struct CapabilityResult {
    pub model: String,
    pub hardware: String,
    pub missing_ops: Vec<String>,
    pub status: CapStatus,
}

// ---------------------------------------------------------------------------
// Fallback recommendations
// ---------------------------------------------------------------------------

/// Suggest a fallback for a missing operation.
pub fn fallback_for_op(op: &str) -> &str {
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

pub fn define_model_architectures() -> Vec<ArchConstraints> {
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

pub fn define_hardware_profiles() -> Vec<HardwareProfile> {
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
pub fn classify_status(total_ops: usize, missing_count: usize) -> CapStatus {
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
pub fn check_capability(arch: &ArchConstraints, hw: &HardwareProfile) -> CapabilityResult {
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
pub fn check_all_capabilities(
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

pub fn print_capability_matrix(
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

pub fn print_failure_details(results: &[CapabilityResult]) {
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
