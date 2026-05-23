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
use super::types::*;

#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

/// Count non-finite values in a weight slice.
pub fn count_nonfinite(weights: &[f64]) -> (usize, usize) {
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

/// Returns `None` only if the layer has zero weights.
pub fn compute_layer_diagnostics(weights: &[f64]) -> Option<LayerDiagnostics> {
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

/// Checks in priority order: NaN, Inf, HighVariance, DeadLayer.
pub fn trace_model(model: &DebugModel, median_std: f64) -> Option<DetectedIssue> {
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
pub fn compute_median_std(model: &DebugModel) -> f64 {
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

/// Replace NaN values with mean of surrounding non-NaN neighbors
pub fn fix_interpolate_nan(weights: &mut [f64]) {
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

/// by the outliers themselves, then clamp.
pub fn fix_clamp_outliers(weights: &mut [f64]) {
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
pub fn apply_fix(model: &mut DebugModel, issue: &DetectedIssue) -> FixAction {
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

/// Run the full 10-stage validation pipeline on the model
pub fn run_validation_pipeline(model: &DebugModel) -> Vec<CheckResult> {
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

/// Run the iterative debug-fix loop on a model
pub fn run_debug_fix_loop(model: &mut DebugModel) -> DebugReport {
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
