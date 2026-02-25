//! # Recipe: Iterative Debug-Fix Loop
//!
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

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

// ============================================================================
// Issue Classification
// ============================================================================

/// Types of model issues detected during tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueType {
    /// NaN values in weights (numerical instability)
    NaN,
    /// Anomalously high variance (outlier weights)
    HighVariance,
    /// Infinite values in weights
    Inf,
    /// Layer with all-zero or near-zero weights
    DeadLayer,
}

impl fmt::Display for IssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaN => write!(f, "NaN"),
            Self::HighVariance => write!(f, "HighVariance"),
            Self::Inf => write!(f, "Inf"),
            Self::DeadLayer => write!(f, "DeadLayer"),
        }
    }
}

/// Actions that can be applied to fix model issues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixAction {
    /// Replace NaN values with interpolated neighbors
    InterpolateNaN,
    /// Clamp outlier weights to [-3*std, +3*std]
    ClampOutliers,
    /// Rescale layer weights to target variance
    RescaleWeights,
    /// Reinitialize dead neurons with small random values
    PruneDeadNeurons,
}

impl fmt::Display for FixAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InterpolateNaN => write!(f, "InterpolateNaN"),
            Self::ClampOutliers => write!(f, "ClampOutliers"),
            Self::RescaleWeights => write!(f, "RescaleWeights"),
            Self::PruneDeadNeurons => write!(f, "PruneDeadNeurons"),
        }
    }
}

// ============================================================================
// Debug Report Structures
// ============================================================================

/// Record of a single debug-fix iteration
#[derive(Debug, Clone)]
pub struct DebugIteration {
    /// Iteration number (1-indexed)
    pub iteration: usize,
    /// Type of issue detected
    pub issue_type: IssueType,
    /// Name of the affected layer
    pub layer_name: String,
    /// Fix action applied
    pub fix_applied: FixAction,
    /// Metric value before the fix
    pub before_value: f64,
    /// Metric value after the fix
    pub after_value: f64,
    /// Whether the issue was resolved
    pub resolved: bool,
}

/// Final report from the debug-fix loop
#[derive(Debug, Clone)]
pub struct DebugReport {
    /// All iterations performed
    pub iterations: Vec<DebugIteration>,
    /// Whether the final validation pipeline passed
    pub final_check_passed: bool,
}

impl DebugReport {
    /// Count how many iterations resolved their issue
    #[must_use]
    pub fn resolved_count(&self) -> usize {
        self.iterations.iter().filter(|i| i.resolved).count()
    }

    /// Return the overall verdict string
    #[must_use]
    pub fn verdict(&self) -> &'static str {
        if self.final_check_passed {
            "FIXED"
        } else {
            "NEEDS_MORE_WORK"
        }
    }
}

// ============================================================================
// Synthetic Model with Layers
// ============================================================================

/// A layer in the synthetic model
#[derive(Debug, Clone)]
struct ModelLayer {
    name: String,
    weights: Vec<f64>,
}

/// A multi-layer synthetic model for debugging
#[derive(Debug, Clone)]
struct DebugModel {
    layers: Vec<ModelLayer>,
}

impl DebugModel {
    /// Build a synthetic model with `n_layers` layers, each having `layer_size` weights.
    /// Uses `ctx.rng()` for deterministic generation.
    fn build(ctx: &mut RecipeContext, n_layers: usize, layer_size: usize) -> Self {
        let rng = ctx.rng();
        let layers = (0..n_layers)
            .map(|i| {
                let weights: Vec<f64> = (0..layer_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
                ModelLayer {
                    name: format!("layer_{i}"),
                    weights,
                }
            })
            .collect();
        Self { layers }
    }

    /// Inject NaN values into a specific layer at given indices
    fn inject_nan(&mut self, layer_idx: usize, indices: &[usize]) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            for &idx in indices {
                if idx < layer.weights.len() {
                    layer.weights[idx] = f64::NAN;
                }
            }
        }
    }

    /// Inject high-variance outliers into a specific layer
    fn inject_outliers(&mut self, layer_idx: usize, indices: &[usize], magnitude: f64) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            for (i, &idx) in indices.iter().enumerate() {
                if idx < layer.weights.len() {
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    layer.weights[idx] = sign * magnitude;
                }
            }
        }
    }
}

// ============================================================================
// Layer Statistics
// ============================================================================

/// Statistics computed from a single layer's weights
#[derive(Debug, Clone)]
struct LayerDiagnostics {
    nan_count: usize,
    inf_count: usize,
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
    dead_ratio: f64,
}

/// Accumulator for finite weight statistics (sum, sum-of-squares, extrema).
struct FiniteStats {
    sum: f64,
    sum_sq: f64,
    count: usize,
    min_val: f64,
    max_val: f64,
    near_zero_count: usize,
}

impl FiniteStats {
    fn new() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            min_val: f64::MAX,
            max_val: f64::MIN,
            near_zero_count: 0,
        }
    }

    fn accumulate(&mut self, w: f64) {
        self.sum += w;
        self.sum_sq += w * w;
        self.count += 1;
        self.min_val = self.min_val.min(w);
        self.max_val = self.max_val.max(w);
        if w.abs() < 1e-10 {
            self.near_zero_count += 1;
        }
    }

    fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    fn std_dev(&self) -> f64 {
        if self.count > 1 {
            let m = self.mean();
            (self.sum_sq / self.count as f64 - m * m).max(0.0).sqrt()
        } else {
            0.0
        }
    }
}

/// Count non-finite values in a weight slice.
fn count_nonfinite(weights: &[f64]) -> (usize, usize) {
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for &w in weights {
        if w.is_nan() {
            nan_count += 1;
        } else if w.is_infinite() {
            inf_count += 1;
        }
    }
    (nan_count, inf_count)
}

/// Compute diagnostics for a layer's weights.
/// Returns `None` only if the layer has zero weights.
fn compute_layer_diagnostics(weights: &[f64]) -> Option<LayerDiagnostics> {
    if weights.is_empty() {
        return None;
    }

    let (nan_count, inf_count) = count_nonfinite(weights);

    let mut stats = FiniteStats::new();
    for &w in weights {
        if w.is_finite() {
            stats.accumulate(w);
        }
    }

    let (min_val, max_val) = if stats.count > 0 {
        (stats.min_val, stats.max_val)
    } else {
        (0.0, 0.0)
    };

    Some(LayerDiagnostics {
        nan_count,
        inf_count,
        mean: stats.mean(),
        std_dev: stats.std_dev(),
        min_val,
        max_val,
        dead_ratio: stats.near_zero_count as f64 / weights.len() as f64,
    })
}

// ============================================================================
// Trace Phase: Detect Issues
// ============================================================================

/// Issue detected during a trace pass
#[derive(Debug, Clone)]
struct DetectedIssue {
    issue_type: IssueType,
    layer_idx: usize,
    layer_name: String,
    metric_value: f64,
}

/// Run a trace pass over all layers, returning the first issue found (if any).
/// Checks in priority order: NaN, Inf, HighVariance, DeadLayer.
fn trace_model(model: &DebugModel, median_std: f64) -> Option<DetectedIssue> {
    for (idx, layer) in model.layers.iter().enumerate() {
        let Some(diag) = compute_layer_diagnostics(&layer.weights) else {
            continue;
        };

        // Priority 1: NaN
        if diag.nan_count > 0 {
            return Some(DetectedIssue {
                issue_type: IssueType::NaN,
                layer_idx: idx,
                layer_name: layer.name.clone(),
                metric_value: diag.nan_count as f64,
            });
        }

        // Priority 2: Inf
        if diag.inf_count > 0 {
            return Some(DetectedIssue {
                issue_type: IssueType::Inf,
                layer_idx: idx,
                layer_name: layer.name.clone(),
                metric_value: diag.inf_count as f64,
            });
        }

        // Priority 3: High variance (std > 10x median std)
        let variance_threshold = if median_std > 0.0 {
            median_std * 10.0
        } else {
            10.0
        };
        if diag.std_dev > variance_threshold {
            return Some(DetectedIssue {
                issue_type: IssueType::HighVariance,
                layer_idx: idx,
                layer_name: layer.name.clone(),
                metric_value: diag.std_dev,
            });
        }

        // Priority 4: Dead layer (>95% near-zero)
        if diag.dead_ratio > 0.95 {
            return Some(DetectedIssue {
                issue_type: IssueType::DeadLayer,
                layer_idx: idx,
                layer_name: layer.name.clone(),
                metric_value: diag.dead_ratio,
            });
        }
    }
    None
}

/// Compute the median standard deviation across all layers
fn compute_median_std(model: &DebugModel) -> f64 {
    let mut stds: Vec<f64> = model
        .layers
        .iter()
        .filter_map(|l| compute_layer_diagnostics(&l.weights))
        .map(|d| d.std_dev)
        .filter(|s| s.is_finite())
        .collect();

    if stds.is_empty() {
        return 0.0;
    }
    stds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = stds.len() / 2;
    if stds.len() % 2 == 0 {
        (stds[mid - 1] + stds[mid]) / 2.0
    } else {
        stds[mid]
    }
}

// ============================================================================
// Fix Phase: Apply Targeted Repairs
// ============================================================================

/// Replace NaN values with mean of surrounding non-NaN neighbors
fn fix_interpolate_nan(weights: &mut [f64]) {
    let len = weights.len();
    let snapshot: Vec<f64> = weights.to_vec();

    for i in 0..len {
        if !snapshot[i].is_nan() {
            continue;
        }
        // Gather non-NaN neighbors (up to 2 on each side)
        let mut neighbor_sum = 0.0_f64;
        let mut neighbor_count = 0usize;
        let start = i.saturating_sub(2);
        let end = (i + 3).min(len);
        for (j, &val) in snapshot.iter().enumerate().take(end).skip(start) {
            if j != i && val.is_finite() {
                neighbor_sum += val;
                neighbor_count += 1;
            }
        }
        weights[i] = if neighbor_count > 0 {
            neighbor_sum / neighbor_count as f64
        } else {
            0.0
        };
    }
}

/// Clamp outlier weights to [-3*std, +3*std] range.
/// Uses a robust two-pass approach: first compute the median and MAD
/// (median absolute deviation) to identify outliers without being skewed
/// by the outliers themselves, then clamp.
fn fix_clamp_outliers(weights: &mut [f64]) {
    let finite_vals: Vec<f64> = weights.iter().copied().filter(|w| w.is_finite()).collect();
    if finite_vals.is_empty() {
        return;
    }

    // Compute median
    let mut sorted = finite_vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Compute MAD (median absolute deviation)
    let mut abs_devs: Vec<f64> = finite_vals.iter().map(|v| (v - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if abs_devs.len() % 2 == 0 {
        (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
    } else {
        abs_devs[abs_devs.len() / 2]
    };

    // Robust std estimate: MAD * 1.4826 (for normal distributions)
    let robust_std = (mad * 1.4826).max(1e-10);
    let bound = 3.0 * robust_std;
    let lower = median - bound;
    let upper = median + bound;

    for w in weights.iter_mut() {
        if w.is_finite() {
            *w = w.clamp(lower, upper);
        }
    }
}

/// Apply the appropriate fix for a detected issue
fn apply_fix(model: &mut DebugModel, issue: &DetectedIssue) -> FixAction {
    let layer = &mut model.layers[issue.layer_idx];
    match issue.issue_type {
        IssueType::NaN => {
            fix_interpolate_nan(&mut layer.weights);
            FixAction::InterpolateNaN
        }
        IssueType::HighVariance => {
            fix_clamp_outliers(&mut layer.weights);
            FixAction::ClampOutliers
        }
        IssueType::Inf => {
            // Replace Inf with 0.0 then interpolate
            for w in &mut layer.weights {
                if w.is_infinite() {
                    *w = f64::NAN;
                }
            }
            fix_interpolate_nan(&mut layer.weights);
            FixAction::RescaleWeights
        }
        IssueType::DeadLayer => {
            // Reinitialize near-zero weights with small random values
            let mut seed: u64 = 0xDEAD_BEEF;
            for w in &mut layer.weights {
                if w.abs() < 1e-10 {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    *w = ((seed >> 33) as f64) / f64::from(u32::MAX) * 0.02 - 0.01;
                }
            }
            FixAction::PruneDeadNeurons
        }
    }
}

// ============================================================================
// Validation Pipeline
// ============================================================================

/// A single validation check result
#[derive(Debug)]
struct CheckResult {
    name: &'static str,
    passed: bool,
}

/// Run the full 10-stage validation pipeline on the model
fn run_validation_pipeline(model: &DebugModel) -> Vec<CheckResult> {
    let median_std = compute_median_std(model);
    let variance_threshold = if median_std > 0.0 {
        median_std * 10.0
    } else {
        10.0
    };

    let all_diags: Vec<Option<LayerDiagnostics>> = model
        .layers
        .iter()
        .map(|l| compute_layer_diagnostics(&l.weights))
        .collect();

    let has_nan = all_diags
        .iter()
        .any(|d| d.as_ref().is_some_and(|d| d.nan_count > 0));
    let has_inf = all_diags
        .iter()
        .any(|d| d.as_ref().is_some_and(|d| d.inf_count > 0));
    let has_high_var = all_diags
        .iter()
        .any(|d| d.as_ref().is_some_and(|d| d.std_dev > variance_threshold));
    let has_dead = all_diags
        .iter()
        .any(|d| d.as_ref().is_some_and(|d| d.dead_ratio > 0.95));

    let weight_range_ok = all_diags
        .iter()
        .all(|d| d.as_ref().map_or(true, |d| d.max_val - d.min_val < 1000.0));
    let mean_centered = all_diags
        .iter()
        .all(|d| d.as_ref().map_or(true, |d| d.mean.abs() < 5.0));
    let std_bounded = all_diags
        .iter()
        .all(|d| d.as_ref().map_or(true, |d| d.std_dev < 50.0));
    let total_params: usize = model.layers.iter().map(|l| l.weights.len()).sum();
    let has_params = total_params > 0;
    let layer_count_ok = model.layers.len() >= 2;
    let all_finite = !has_nan && !has_inf;

    vec![
        CheckResult {
            name: "no_nan",
            passed: !has_nan,
        },
        CheckResult {
            name: "no_inf",
            passed: !has_inf,
        },
        CheckResult {
            name: "no_high_variance",
            passed: !has_high_var,
        },
        CheckResult {
            name: "no_dead_layers",
            passed: !has_dead,
        },
        CheckResult {
            name: "weight_range_bounded",
            passed: weight_range_ok,
        },
        CheckResult {
            name: "mean_centered",
            passed: mean_centered,
        },
        CheckResult {
            name: "std_bounded",
            passed: std_bounded,
        },
        CheckResult {
            name: "has_parameters",
            passed: has_params,
        },
        CheckResult {
            name: "min_layer_count",
            passed: layer_count_ok,
        },
        CheckResult {
            name: "all_finite",
            passed: all_finite,
        },
    ]
}

// ============================================================================
// Debug-Fix Loop Core
// ============================================================================

/// Maximum number of iterations to prevent infinite loops
const MAX_ITERATIONS: usize = 10;

/// Run the iterative debug-fix loop on a model
fn run_debug_fix_loop(model: &mut DebugModel) -> DebugReport {
    let mut iterations = Vec::new();

    for iter_num in 1..=MAX_ITERATIONS {
        let median_std = compute_median_std(model);
        let Some(issue) = trace_model(model, median_std) else {
            break;
        };

        let before_value = issue.metric_value;
        let layer_name = issue.layer_name.clone();
        let issue_type = issue.issue_type;

        let fix_applied = apply_fix(model, &issue);

        // Re-check the specific layer
        let after_diag = compute_layer_diagnostics(&model.layers[issue.layer_idx].weights);
        let after_value = match issue_type {
            IssueType::NaN => after_diag.map_or(0.0, |d| d.nan_count as f64),
            IssueType::Inf => after_diag.map_or(0.0, |d| d.inf_count as f64),
            IssueType::HighVariance => after_diag.map_or(0.0, |d| d.std_dev),
            IssueType::DeadLayer => after_diag.map_or(0.0, |d| d.dead_ratio),
        };

        let resolved = match issue_type {
            IssueType::NaN => after_value < 1.0,
            IssueType::Inf => after_value < 1.0,
            IssueType::HighVariance => {
                let new_median = compute_median_std(model);
                let threshold = if new_median > 0.0 {
                    new_median * 10.0
                } else {
                    10.0
                };
                after_value <= threshold
            }
            IssueType::DeadLayer => after_value <= 0.95,
        };

        iterations.push(DebugIteration {
            iteration: iter_num,
            issue_type,
            layer_name,
            fix_applied,
            before_value,
            after_value,
            resolved,
        });
    }

    let checks = run_validation_pipeline(model);
    let final_check_passed = checks.iter().all(|c| c.passed);

    DebugReport {
        iterations,
        final_check_passed,
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

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
