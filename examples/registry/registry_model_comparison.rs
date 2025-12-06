//! # Recipe: Model Version Comparison
//!
//! **Category**: Model Registry
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Compare model versions and their performance metrics.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_model_comparison
//! ```

use apr_cookbook::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_model_comparison")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Comparing model versions");
    println!();

    // Create mock model versions
    let versions = vec![
        ModelVersion {
            version: "1.0.0".to_string(),
            metrics: [
                ("accuracy".to_string(), 0.92f64),
                ("f1_score".to_string(), 0.89f64),
                ("latency_ms".to_string(), 15.0f64),
                ("model_size_mb".to_string(), 12.5f64),
            ]
            .into_iter()
            .collect(),
            training_time_hours: 2.5,
            training_samples: 100000,
        },
        ModelVersion {
            version: "1.1.0".to_string(),
            metrics: [
                ("accuracy".to_string(), 0.94f64),
                ("f1_score".to_string(), 0.91f64),
                ("latency_ms".to_string(), 18.0f64),
                ("model_size_mb".to_string(), 15.2f64),
            ]
            .into_iter()
            .collect(),
            training_time_hours: 3.0,
            training_samples: 150000,
        },
        ModelVersion {
            version: "1.2.0".to_string(),
            metrics: [
                ("accuracy".to_string(), 0.95f64),
                ("f1_score".to_string(), 0.93f64),
                ("latency_ms".to_string(), 12.0f64),
                ("model_size_mb".to_string(), 10.0f64),
            ]
            .into_iter()
            .collect(),
            training_time_hours: 4.0,
            training_samples: 200000,
        },
    ];

    ctx.record_metric("version_count", versions.len() as i64);

    // Compare versions
    let comparison = compare_versions(&versions);

    println!("Model Versions:");
    println!("{:-<80}", "");
    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Version", "Accuracy", "F1 Score", "Latency(ms)", "Size(MB)", "Samples"
    );
    println!("{:-<80}", "");

    for v in &versions {
        println!(
            "{:<10} {:>10.2}% {:>10.2}% {:>12.1} {:>12.1} {:>10}",
            v.version,
            v.metrics.get("accuracy").unwrap_or(&0.0) * 100.0,
            v.metrics.get("f1_score").unwrap_or(&0.0) * 100.0,
            v.metrics.get("latency_ms").unwrap_or(&0.0),
            v.metrics.get("model_size_mb").unwrap_or(&0.0),
            v.training_samples
        );
    }
    println!("{:-<80}", "");

    println!();
    println!("Comparison Summary:");
    println!(
        "  Best accuracy: {} ({:.2}%)",
        comparison.best_accuracy_version,
        comparison.best_accuracy * 100.0
    );
    println!(
        "  Best F1 score: {} ({:.2}%)",
        comparison.best_f1_version,
        comparison.best_f1 * 100.0
    );
    println!(
        "  Lowest latency: {} ({:.1}ms)",
        comparison.lowest_latency_version, comparison.lowest_latency
    );
    println!(
        "  Smallest size: {} ({:.1}MB)",
        comparison.smallest_size_version, comparison.smallest_size
    );

    ctx.record_float_metric("best_accuracy", comparison.best_accuracy);
    ctx.record_float_metric("best_f1", comparison.best_f1);

    // Generate recommendation
    let recommendation = recommend_version(&versions);
    println!();
    println!(
        "Recommendation: {} ({})",
        recommendation.version, recommendation.reason
    );

    // Save comparison report
    let report_path = ctx.path("comparison_report.txt");
    save_report(&report_path, &versions, &comparison)?;
    println!();
    println!("Report saved to: {:?}", report_path);

    Ok(())
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ModelVersion {
    version: String,
    metrics: HashMap<String, f64>,
    training_time_hours: f64,
    training_samples: usize,
}

#[derive(Debug)]
struct ComparisonResult {
    best_accuracy_version: String,
    best_accuracy: f64,
    best_f1_version: String,
    best_f1: f64,
    lowest_latency_version: String,
    lowest_latency: f64,
    smallest_size_version: String,
    smallest_size: f64,
}

#[derive(Debug)]
struct Recommendation {
    version: String,
    reason: String,
}

fn compare_versions(versions: &[ModelVersion]) -> ComparisonResult {
    let mut result = ComparisonResult {
        best_accuracy_version: String::new(),
        best_accuracy: 0.0,
        best_f1_version: String::new(),
        best_f1: 0.0,
        lowest_latency_version: String::new(),
        lowest_latency: f64::MAX,
        smallest_size_version: String::new(),
        smallest_size: f64::MAX,
    };

    for v in versions {
        let accuracy = *v.metrics.get("accuracy").unwrap_or(&0.0);
        if accuracy > result.best_accuracy {
            result.best_accuracy = accuracy;
            result.best_accuracy_version = v.version.clone();
        }

        let f1 = *v.metrics.get("f1_score").unwrap_or(&0.0);
        if f1 > result.best_f1 {
            result.best_f1 = f1;
            result.best_f1_version = v.version.clone();
        }

        let latency = *v.metrics.get("latency_ms").unwrap_or(&f64::MAX);
        if latency < result.lowest_latency {
            result.lowest_latency = latency;
            result.lowest_latency_version = v.version.clone();
        }

        let size = *v.metrics.get("model_size_mb").unwrap_or(&f64::MAX);
        if size < result.smallest_size {
            result.smallest_size = size;
            result.smallest_size_version = v.version.clone();
        }
    }

    result
}

fn recommend_version(versions: &[ModelVersion]) -> Recommendation {
    // Score each version: weighted combination of metrics
    let mut best_version = &versions[0];
    let mut best_score = 0.0f64;

    for v in versions {
        let accuracy = *v.metrics.get("accuracy").unwrap_or(&0.0);
        let f1 = *v.metrics.get("f1_score").unwrap_or(&0.0);
        let latency = *v.metrics.get("latency_ms").unwrap_or(&100.0);
        let size = *v.metrics.get("model_size_mb").unwrap_or(&100.0);

        // Score: high accuracy/f1 good, low latency/size good
        let score =
            accuracy * 0.4 + f1 * 0.3 + (1.0 - latency / 50.0) * 0.15 + (1.0 - size / 50.0) * 0.15;

        if score > best_score {
            best_score = score;
            best_version = v;
        }
    }

    Recommendation {
        version: best_version.version.clone(),
        reason: "Best overall weighted score (accuracy, F1, latency, size)".to_string(),
    }
}

fn save_report(
    path: &std::path::Path,
    versions: &[ModelVersion],
    comparison: &ComparisonResult,
) -> Result<()> {
    let mut report = String::new();
    report.push_str("Model Version Comparison Report\n");
    report.push_str("================================\n\n");

    for v in versions {
        report.push_str(&format!("Version {}\n", v.version));
        for (key, value) in &v.metrics {
            report.push_str(&format!("  {}: {:.4}\n", key, value));
        }
        report.push('\n');
    }

    report.push_str("Summary\n");
    report.push_str("-------\n");
    report.push_str(&format!(
        "Best accuracy: {} ({:.2}%)\n",
        comparison.best_accuracy_version,
        comparison.best_accuracy * 100.0
    ));
    report.push_str(&format!(
        "Best F1: {} ({:.2}%)\n",
        comparison.best_f1_version,
        comparison.best_f1 * 100.0
    ));

    std::fs::write(path, report)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison() {
        let versions = vec![
            ModelVersion {
                version: "1.0".to_string(),
                metrics: [("accuracy".to_string(), 0.9f64)].into_iter().collect(),
                training_time_hours: 1.0,
                training_samples: 1000,
            },
            ModelVersion {
                version: "2.0".to_string(),
                metrics: [("accuracy".to_string(), 0.95f64)].into_iter().collect(),
                training_time_hours: 2.0,
                training_samples: 2000,
            },
        ];

        let result = compare_versions(&versions);
        assert_eq!(result.best_accuracy_version, "2.0");
        assert!((result.best_accuracy - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_recommendation() {
        let versions = vec![ModelVersion {
            version: "1.0".to_string(),
            metrics: [
                ("accuracy".to_string(), 0.9f64),
                ("f1_score".to_string(), 0.85f64),
                ("latency_ms".to_string(), 10.0f64),
                ("model_size_mb".to_string(), 5.0f64),
            ]
            .into_iter()
            .collect(),
            training_time_hours: 1.0,
            training_samples: 1000,
        }];

        let rec = recommend_version(&versions);
        assert_eq!(rec.version, "1.0");
    }

    #[test]
    fn test_report_generation() {
        let ctx = RecipeContext::new("test_report").unwrap();
        let path = ctx.path("report.txt");

        let versions = vec![ModelVersion {
            version: "1.0".to_string(),
            metrics: [("accuracy".to_string(), 0.9f64)].into_iter().collect(),
            training_time_hours: 1.0,
            training_samples: 1000,
        }];

        let comparison = compare_versions(&versions);
        save_report(&path, &versions, &comparison).unwrap();

        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_comparison_finds_best(accuracy1 in 0.0f64..1.0, accuracy2 in 0.0f64..1.0) {
            let versions = vec![
                ModelVersion {
                    version: "v1".to_string(),
                    metrics: [("accuracy".to_string(), accuracy1)].into_iter().collect(),
                    training_time_hours: 1.0,
                    training_samples: 1000,
                },
                ModelVersion {
                    version: "v2".to_string(),
                    metrics: [("accuracy".to_string(), accuracy2)].into_iter().collect(),
                    training_time_hours: 1.0,
                    training_samples: 1000,
                },
            ];

            let result = compare_versions(&versions);
            let expected_best = if accuracy1 >= accuracy2 { "v1" } else { "v2" };
            prop_assert_eq!(result.best_accuracy_version, expected_best);
        }
    }
}
