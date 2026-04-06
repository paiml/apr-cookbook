#![allow(unused_imports)]
//! # Recipe: Iterative Debug-Fix Loop
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Advanced - Model Debugging & Repair
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//!
//! Demonstrates an iterative debug-fix loop that composes:
//! trace -> debug -> fix -> check -> validate
//!
//! Each iteration identifies a model issue, diagnoses the root cause,
//! applies a targeted fix, and verifies the repair. The loop continues
//! until all issues are resolved or a maximum iteration count is reached.
//!
//! ## Architecture
//!
//! ```text
//! Model -> Trace (layer-by-layer) -> Detect Issue
//!                                        |
//!                          Debug (inspect layer stats)
//!                                        |
//!                          Fix (apply targeted repair)
//!                                        |
//!                          Check (verify fix worked)
//!                                        |
//!                          Validate (full pipeline check)
//! ```
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic defect detection at each layer
//! - **Kaizen**: Iterative improvement until all issues resolved
//! - **Genchi Genbutsu**: Inspect actual weight values, not abstractions
//! - **Poka-yoke**: Validation gate prevents deploying broken models
//!
//! ## Run Command
//! ```bash
//! cargo run --example debug_fix_loop
//! ```
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
use rand::Rng;
use std::fmt;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!("================================================================");
    println!("  Iterative Debug-Fix Loop");
    println!("  Toyota Way: Jidoka + Kaizen (detect, fix, repeat)");
    println!("================================================================");
    println!();

    let mut ctx = RecipeContext::new("debug_fix_loop")?;

    // Section 1: Build synthetic model with known issues
    println!("--- Section 1: Build synthetic model (10 layers x 256 weights) ---");
    let mut model = DebugModel::build(&mut ctx, 10, 256);
    println!("  Built model with {} layers", model.layers.len());

    // Inject NaN in layer 3 (indices 10, 50, 100)
    model.inject_nan(3, &[10, 50, 100]);
    println!("  Injected 3 NaN values into layer_3");

    // Inject high-variance outliers in layer 7 (magnitude 500.0)
    model.inject_outliers(7, &[0, 20, 40, 60, 80], 500.0);
    println!("  Injected 5 outliers (magnitude 500) into layer_7");
    println!();

    // Section 2: Run the debug-fix loop
    println!("--- Section 2: Run debug-fix loop ---");
    let report = run_debug_fix_loop(&mut model);

    // Section 3: Print iteration log
    println!();
    println!("--- Section 3: Iteration Log ---");
    println!(
        "{:<5} {:<15} {:<10} {:<20} {:<12} {:<12} {:<8}",
        "Iter", "Issue", "Layer", "Fix", "Before", "After", "OK?"
    );
    println!("{}", "-".repeat(82));
    for it in &report.iterations {
        println!(
            "{:<5} {:<15} {:<10} {:<20} {:<12.4} {:<12.4} {:<8}",
            it.iteration,
            it.issue_type.to_string(),
            it.layer_name,
            it.fix_applied.to_string(),
            it.before_value,
            it.after_value,
            if it.resolved { "YES" } else { "NO" },
        );
    }
    println!();

    // Section 4: Final validation
    println!("--- Section 4: Final Validation (10-stage pipeline) ---");
    let checks = run_validation_pipeline(&model);
    for check in &checks {
        let mark = if check.passed { "PASS" } else { "FAIL" };
        println!("  [{mark}] {}", check.name);
    }
    println!();

    let all_passed = checks.iter().all(|c| c.passed);
    println!(
        "Verdict: {} ({} iterations, {}/{} resolved)",
        if all_passed {
            "FIXED"
        } else {
            "NEEDS_MORE_WORK"
        },
        report.iterations.len(),
        report.resolved_count(),
        report.iterations.len(),
    );

    ctx.record_metric("iterations", report.iterations.len() as i64);
    ctx.record_metric("resolved", report.resolved_count() as i64);
    ctx.record_metric("final_passed", i64::from(report.final_check_passed));

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_detection_and_fix() {
        let mut ctx = RecipeContext::new("test_nan_fix").expect("context");
        let mut model = DebugModel::build(&mut ctx, 5, 64);
        model.inject_nan(2, &[5, 10, 15]);

        let median_std = compute_median_std(&model);
        let issue = trace_model(&model, median_std);
        assert!(issue.is_some());
        let issue = issue.expect("should detect NaN");
        assert_eq!(issue.issue_type, IssueType::NaN);
        assert_eq!(issue.layer_idx, 2);
        assert!((issue.metric_value - 3.0).abs() < f64::EPSILON);

        apply_fix(&mut model, &issue);
        let diag = compute_layer_diagnostics(&model.layers[2].weights).expect("diag");
        assert_eq!(diag.nan_count, 0, "NaN should be fixed");
    }

    #[test]
    fn test_high_variance_detection_and_fix() {
        let mut ctx = RecipeContext::new("test_var_fix").expect("context");
        let mut model = DebugModel::build(&mut ctx, 8, 128);
        model.inject_outliers(5, &[0, 10, 20], 1000.0);

        let before_diag = compute_layer_diagnostics(&model.layers[5].weights).expect("before diag");
        let median_std = compute_median_std(&model);
        let issue = trace_model(&model, median_std);
        assert!(issue.is_some());
        let issue = issue.expect("should detect high variance");
        assert_eq!(issue.issue_type, IssueType::HighVariance);
        assert_eq!(issue.layer_idx, 5);

        apply_fix(&mut model, &issue);
        let after_diag = compute_layer_diagnostics(&model.layers[5].weights).expect("after diag");
        assert!(
            after_diag.std_dev < before_diag.std_dev,
            "Std dev should be reduced after clamping: before={}, after={}",
            before_diag.std_dev,
            after_diag.std_dev
        );
    }

    #[test]
    fn test_full_loop_resolves_all_issues() {
        let mut ctx = RecipeContext::new("test_full_loop").expect("context");
        let mut model = DebugModel::build(&mut ctx, 10, 256);
        model.inject_nan(3, &[10, 50, 100]);
        model.inject_outliers(7, &[0, 20, 40, 60, 80], 500.0);

        let report = run_debug_fix_loop(&mut model);
        assert!(
            report.final_check_passed,
            "All issues should be resolved: {:?}",
            report
                .iterations
                .iter()
                .map(|i| format!("{}:{}", i.issue_type, i.resolved))
                .collect::<Vec<_>>()
        );
        assert!(
            report.iterations.len() >= 2,
            "Should have at least 2 iterations (NaN + HighVariance)"
        );
    }

    #[test]
    fn test_clean_model_no_iterations() {
        let mut ctx = RecipeContext::new("test_clean").expect("context");
        let mut model = DebugModel::build(&mut ctx, 5, 64);

        let report = run_debug_fix_loop(&mut model);
        assert!(report.iterations.is_empty(), "Clean model needs no fixes");
        assert!(report.final_check_passed);
    }

    #[test]
    fn test_validation_pipeline_healthy_model() {
        let mut ctx = RecipeContext::new("test_validation").expect("context");
        let model = DebugModel::build(&mut ctx, 4, 32);

        let checks = run_validation_pipeline(&model);
        assert_eq!(checks.len(), 10, "Pipeline should have 10 stages");
        for check in &checks {
            assert!(
                check.passed,
                "Check '{}' should pass on healthy model",
                check.name
            );
        }
    }

    #[test]
    fn test_validation_detects_nan() {
        let mut ctx = RecipeContext::new("test_val_nan").expect("context");
        let mut model = DebugModel::build(&mut ctx, 4, 32);
        model.inject_nan(0, &[0]);

        let checks = run_validation_pipeline(&model);
        let nan_check = checks
            .iter()
            .find(|c| c.name == "no_nan")
            .expect("no_nan check");
        assert!(!nan_check.passed, "Should detect NaN");
        let finite_check = checks
            .iter()
            .find(|c| c.name == "all_finite")
            .expect("all_finite check");
        assert!(!finite_check.passed, "Should detect non-finite");
    }

    #[test]
    fn test_interpolate_nan_neighbors() {
        let mut weights = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        fix_interpolate_nan(&mut weights);
        // NaN at index 2 should be mean of neighbors: (1.0 + 2.0 + 4.0 + 5.0) / 4 = 3.0
        assert!(
            (weights[2] - 3.0).abs() < 1e-10,
            "NaN should be interpolated to 3.0, got {}",
            weights[2]
        );
    }

    #[test]
    fn test_interpolate_nan_edge() {
        let mut weights = vec![f64::NAN, 2.0, 3.0];
        fix_interpolate_nan(&mut weights);
        // NaN at index 0: neighbors are index 1 (2.0) and index 2 (3.0)
        assert!(
            (weights[0] - 2.5).abs() < 1e-10,
            "Edge NaN should be interpolated to 2.5, got {}",
            weights[0]
        );
    }

    #[test]
    fn test_clamp_outliers_bounds() {
        let mut weights = vec![0.1, 0.2, 0.15, 0.12, 100.0, -100.0];
        fix_clamp_outliers(&mut weights);
        let diag = compute_layer_diagnostics(&weights).expect("diag");
        // After clamping, the extreme values should be reduced
        assert!(diag.max_val < 100.0, "Max should be reduced after clamping");
        assert!(
            diag.min_val > -100.0,
            "Min should be increased after clamping"
        );
    }

    #[test]
    fn test_debug_iteration_fields() {
        let it = DebugIteration {
            iteration: 1,
            issue_type: IssueType::NaN,
            layer_name: "layer_3".to_string(),
            fix_applied: FixAction::InterpolateNaN,
            before_value: 3.0,
            after_value: 0.0,
            resolved: true,
        };
        assert_eq!(it.iteration, 1);
        assert_eq!(it.issue_type, IssueType::NaN);
        assert_eq!(it.fix_applied, FixAction::InterpolateNaN);
        assert!(it.resolved);
    }

    #[test]
    fn test_debug_report_verdict() {
        let report = DebugReport {
            iterations: vec![
                DebugIteration {
                    iteration: 1,
                    issue_type: IssueType::NaN,
                    layer_name: "layer_0".to_string(),
                    fix_applied: FixAction::InterpolateNaN,
                    before_value: 2.0,
                    after_value: 0.0,
                    resolved: true,
                },
                DebugIteration {
                    iteration: 2,
                    issue_type: IssueType::HighVariance,
                    layer_name: "layer_5".to_string(),
                    fix_applied: FixAction::ClampOutliers,
                    before_value: 50.0,
                    after_value: 1.2,
                    resolved: true,
                },
            ],
            final_check_passed: true,
        };
        assert_eq!(report.verdict(), "FIXED");
        assert_eq!(report.resolved_count(), 2);

        let failing = DebugReport {
            iterations: vec![],
            final_check_passed: false,
        };
        assert_eq!(failing.verdict(), "NEEDS_MORE_WORK");
    }
}
