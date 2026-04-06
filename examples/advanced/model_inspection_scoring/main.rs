#![allow(unused_imports)]
//! # Recipe: Model Inspection & Quality Scoring
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Advanced - Observability & Quality
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## 25-Point QA Checklist
//! 1. [x] Build succeeds (`cargo build --release`)
//! 2. [x] Tests pass (`cargo test`)
//! 3. [x] Clippy clean (`cargo clippy -- -D warnings`)
//! 4. [x] Format clean (`cargo fmt --check`)
//! 5. [x] Documentation >90% coverage
//! 6. [x] Unit test coverage >95%
//! 7. [x] Property tests (100+ cases)
//! 8. [x] No `unwrap()` in logic paths
//! 9. [x] Error handling with `?` or `expect()`
//! 10. [x] Deterministic output (3 runs match)
//! 11. [x] Detects NaN weights (inject test)
//! 12. [x] Detects Inf weights (inject test)
//! 13. [x] Checksum validation (tamper test)
//! 14. [x] Signature validation (invalid sig test)
//! 15. [x] Score accuracy ±2pts (golden models)
//! 16. [x] Diff detects changes (modified model)
//! 17. [x] JSON output valid (schema validation)
//! 18. [x] Human-readable output (manual review)
//! 19. [x] Large model handling (1GB+ model test)
//! 20. [x] Memory-mapped inspection (<100MB overhead)
//! 21. [x] IIUR compliance (isolation test)
//! 22. [x] Toyota Way documented (README)
//! 23. [x] CI integration (Actions pass)
//! 24. [x] Example models included (3 test models)
//! 25. [x] Security audit clean (`cargo audit`)
//!
//! ## Learning Objective
//! Comprehensive model inspection: header parsing, metadata extraction,
//! weight statistics, health scoring, and model comparison (diff).
//!
//! ## Run Command
//! ```bash
//! cargo run --example model_inspection_scoring
//! cargo run --example model_inspection_scoring -- --json
//! ```
//!
//! ## Toyota Way Principles
//! - **Genchi Genbutsu** (Go and see): Direct inspection of model internals
//! - **Jidoka** (Quality built-in): 100-point scoring framework
//! - **Poka-yoke** (Error-proofing): NaN/Inf detection, checksum validation
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::f32;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

mod helpers;
#[allow(unused_imports, clippy::wildcard_imports)]
use helpers::*;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let json_output = args.iter().any(|a| a == "--json" || a == "-j");
    let diff_mode = args.iter().any(|a| a == "--diff" || a == "-d");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Model Inspection & Quality Scoring (Demo C)            ║");
    println!("║       Toyota Way: Genchi Genbutsu (Go and See)               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create recipe context for deterministic execution
    let mut ctx = RecipeContext::new("model_inspection_scoring")?;

    if diff_mode {
        // Demonstrate model diff
        let diff = demonstrate_model_diff(&ctx)?;
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&diff).unwrap_or_default()
            );
        } else {
            print_diff_report(&diff);
        }
    } else {
        // Demonstrate model inspection
        let result = demonstrate_inspection(&ctx)?;
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
        } else {
            print_inspection_report(&result);
        }
    }

    ctx.record_metric("inspection_complete", 1);
    println!("\n✅ Model inspection complete!");
    Ok(())
}

/// Print human-readable inspection report
fn print_inspection_report(result: &InspectionResult) {
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                    INSPECTION REPORT                        │");
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n📋 Header Information:");
    println!("   Magic:            {}", result.header.magic);
    println!(
        "   Version:          {}.{}",
        result.header.version.0, result.header.version.1
    );
    println!(
        "   Compression:      {:.2}x",
        result.header.compression_ratio
    );
    println!("   Checksum:         0x{:08X}", result.header.checksum);

    println!("\n🏷️  Metadata:");
    println!("   Model Type:       {}", result.metadata.model_type);
    println!("   Parameters:       {}", result.metadata.parameters);
    println!("   Framework:        {}", result.metadata.framework);

    println!("\n📊 Weight Statistics:");
    for stats in &result.weight_stats {
        println!("   Layer: {}", stats.name);
        println!("   Shape: {:?}", stats.shape);
        println!("   Range: [{:.6}, {:.6}]", stats.min, stats.max);
        println!("   Mean:  {:.6}, Std: {:.6}", stats.mean, stats.std);
        println!(
            "   NaN: {}, Inf: {}, Zero: {}",
            stats.nan_count, stats.inf_count, stats.zero_count
        );
        println!("   Sparsity: {:.2}%", stats.sparsity * 100.0);
    }

    println!("\n🎯 Quality Score:");
    println!(
        "   Structural:       {}/25",
        result.quality_score.structural
    );
    println!("   Numerical:        {}/25", result.quality_score.numerical);
    println!(
        "   Compression:      {}/25",
        result.quality_score.compression
    );
    println!("   Security:         {}/25", result.quality_score.security);
    println!("   ─────────────────────────");
    println!(
        "   TOTAL:            {}/100 (Grade: {})",
        result.quality_score.total, result.quality_score.grade
    );

    let status_emoji = match result.health_status {
        HealthStatus::Healthy => "✅",
        HealthStatus::Warning => "⚠️",
        HealthStatus::Critical => "❌",
    };
    println!(
        "\n🏥 Health Status: {} {:?}",
        status_emoji, result.health_status
    );
}

/// Print human-readable diff report
fn print_diff_report(diff: &ModelDiff) {
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│                      MODEL DIFF REPORT                      │");
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n📁 Comparing:");
    println!("   Model A: {}", diff.model_a);
    println!("   Model B: {}", diff.model_b);

    println!("\n📊 Overall Metrics:");
    println!("   L2 Distance:      {:.6}", diff.total_l2_distance);
    println!("   Cosine Similarity: {:.6}", diff.cosine_similarity);

    let drift_emoji = if diff.drift_detected { "⚠️" } else { "✅" };
    println!(
        "\n🔍 Drift Detection: {} {}",
        drift_emoji,
        if diff.drift_detected {
            "DRIFT DETECTED"
        } else {
            "No significant drift"
        }
    );

    println!("\n📋 Layer-by-Layer:");
    for layer in &diff.layer_diffs {
        println!("   {} (changed: {})", layer.name, layer.changed);
        println!(
            "     L2: {:.6}, Cosine: {:.6}",
            layer.l2_distance, layer.cosine_similarity
        );
        println!("     Max abs diff: {:.6}", layer.max_abs_diff);
    }
}

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests;

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod proptests;
