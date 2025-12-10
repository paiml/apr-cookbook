//! # Recipe: Inference Explainability
//!
//! **Category**: Inference Monitoring
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: inference-monitoring feature (aprender)
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
//! Demonstrate explainable inference using aprender's Explainable trait.
//! Shows feature contributions and decision paths for model predictions.
//!
//! ## Toyota Way: 現地現物 (Genchi Genbutsu)
//! Every prediction can be traced to its decision path. Go and see.
//!
//! ## Run Command
//! ```bash
//! cargo run --example inference_explainability
//! ```

use apr_cookbook::prelude::*;
use aprender::explainable::IntoExplainable;
use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::Estimator;
use entrenar::monitor::inference::{
    path::{DecisionPath, LinearPath},
    InferenceMonitor, RingCollector, TraceCollector,
};
use serde::Serialize;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_explainability")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Demonstrating explainable inference with decision paths");
    println!();

    // Create and train a simple linear model
    println!("1. Training linear regression model...");
    let (model, feature_names) = train_model()?;
    let explainable = model.into_explainable();

    ctx.record_metric("n_features", feature_names.len() as i64);

    // Create monitored inference with ring buffer collector
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(explainable, collector);

    println!("2. Running monitored inference...");
    println!();

    // Sample data for prediction
    let samples: &[&[f32]] = &[
        &[25.0, 50000.0, 3.0],  // Young, medium income, few accounts
        &[45.0, 120000.0, 7.0], // Middle-aged, high income, many accounts
        &[35.0, 75000.0, 5.0],  // Middle-aged, medium income
    ];

    for (i, sample) in samples.iter().enumerate() {
        println!("--- Sample {} ---", i + 1);
        println!(
            "  Features: {}={}, {}={}, {}={}",
            feature_names[0], sample[0], feature_names[1], sample[1], feature_names[2], sample[2]
        );

        // Make prediction with explanation
        let output = monitor.predict(sample, 1);
        let prediction = output[0];

        println!("  Prediction: {:.4}", prediction);

        // Get the decision path
        let traces = monitor.collector().recent(1);
        if let Some(trace) = traces.first() {
            let path = &trace.path;

            println!("  Confidence: {:.1}%", path.confidence() * 100.0);
            println!();
            println!("  Feature Contributions:");

            let contributions = path.feature_contributions();
            for (j, &contrib) in contributions.iter().enumerate() {
                let sign = if contrib >= 0.0 { "+" } else { "" };
                println!("    - {}: {}{:.4}", feature_names[j], sign, contrib);
            }

            println!();
            println!("  Explanation:");
            println!("  {}", indent_text(&path.explain(), 2));
        }
        println!();

        ctx.record_metric(&format!("prediction_{}", i), (prediction * 1000.0) as i64);
    }

    // Save audit trail
    let audit_path = ctx.path("inference_audit.json");
    save_audit_trail(&audit_path, monitor.collector())?;
    println!("Audit trail saved to: {:?}", audit_path);

    ctx.record_metric("total_inferences", monitor.collector().len() as i64);

    Ok(())
}

/// Train a simple linear regression model for credit scoring
fn train_model() -> Result<(LinearRegression, Vec<String>)> {
    let feature_names = vec![
        "age".to_string(),
        "income".to_string(),
        "num_accounts".to_string(),
    ];

    // Training data: features are age, income (scaled), num_accounts
    // Target: credit score (scaled 0-1)
    let x = Matrix::from_vec(
        5,
        3,
        vec![
            30.0, 60000.0, 2.0, // Sample 1
            40.0, 90000.0, 5.0, // Sample 2
            25.0, 40000.0, 1.0, // Sample 3
            50.0, 150000.0, 8.0, // Sample 4
            35.0, 70000.0, 4.0, // Sample 5
        ],
    )
    .map_err(|e| CookbookError::Aprender(e.to_string()))?;

    let y = Vector::from_slice(&[0.6, 0.75, 0.45, 0.9, 0.65]);

    let mut model = LinearRegression::new();
    model
        .fit(&x, &y)
        .map_err(|e| CookbookError::Aprender(e.to_string()))?;

    println!("   Coefficients: {:?}", model.coefficients().as_slice());
    println!("   Intercept: {:.4}", model.intercept());

    Ok((model, feature_names))
}

/// Indent text by a number of spaces
fn indent_text(text: &str, spaces: usize) -> String {
    let indent = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Save audit trail to JSON file
fn save_audit_trail(
    path: &std::path::Path,
    collector: &RingCollector<LinearPath, 64>,
) -> Result<()> {
    #[derive(Serialize)]
    struct AuditEntry<'a> {
        sequence: u64,
        timestamp_ns: u64,
        path: &'a LinearPath,
        output: f32,
    }

    let entries: Vec<AuditEntry<'_>> = collector
        .recent(collector.len())
        .iter()
        .map(|trace| AuditEntry {
            sequence: trace.sequence,
            timestamp_ns: trace.timestamp_ns,
            path: &trace.path,
            output: trace.output,
        })
        .collect();

    let json = serde_json::to_string_pretty(&entries)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use entrenar::monitor::inference::Explainable;

    #[test]
    fn test_model_training() {
        let result = train_model();
        assert!(result.is_ok());

        let (model, names) = result.unwrap();
        assert_eq!(names.len(), 3);
        assert_eq!(model.coefficients().as_slice().len(), 3);
    }

    #[test]
    fn test_explainable_prediction() {
        let (model, _) = train_model().unwrap();
        let explainable = model.into_explainable();

        let sample = vec![35.0, 80000.0, 4.0];
        let (outputs, paths) = explainable.predict_explained(&sample, 1);

        assert_eq!(outputs.len(), 1);
        assert_eq!(paths.len(), 1);

        let path = &paths[0];
        assert_eq!(path.feature_contributions().len(), 3);
        assert!(path.confidence() > 0.0);
    }

    #[test]
    fn test_monitored_inference() {
        let (model, _) = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: RingCollector<LinearPath, 32> = RingCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        let sample = vec![30.0, 60000.0, 3.0];
        let _ = monitor.predict(&sample, 1);

        assert_eq!(monitor.collector().len(), 1);
    }

    #[test]
    fn test_audit_trail_persistence() {
        let ctx = RecipeContext::new("test_audit").unwrap();
        let path = ctx.path("audit.json");

        let (model, _) = train_model().unwrap();
        let explainable = model.into_explainable();

        let collector: RingCollector<LinearPath, 64> = RingCollector::new();
        let mut monitor = InferenceMonitor::new(explainable, collector);

        let sample = vec![40.0, 100000.0, 5.0];
        let _ = monitor.predict(&sample, 1);

        save_audit_trail(&path, monitor.collector()).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("sequence"));
        assert!(content.contains("timestamp_ns"));
    }

    #[test]
    fn test_explanation_contains_feature_info() {
        let (model, _) = train_model().unwrap();
        let explainable = model.into_explainable();

        let sample = vec![35.0, 80000.0, 4.0];
        let path = explainable.explain_one(&sample);

        let explanation = path.explain();
        assert!(explanation.contains("Prediction"));
        assert!(explanation.contains("feature"));
    }

    #[test]
    fn test_indent_text() {
        let text = "line1\nline2";
        let indented = indent_text(text, 2);
        assert_eq!(indented, "  line1\n  line2");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use entrenar::monitor::inference::Explainable;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_contributions_sum_to_logit(
            age in 18.0f32..80.0,
            income in 20000.0f32..200000.0,
            accounts in 0.0f32..20.0
        ) {
            let (model, _) = train_model().unwrap();
            let intercept = model.intercept();
            let explainable = model.into_explainable();

            let sample = vec![age, income, accounts];
            let path = explainable.explain_one(&sample);

            let contributions: f32 = path.feature_contributions().iter().sum();
            let expected = contributions + intercept;

            // The logit should equal contributions + intercept
            prop_assert!((path.logit - expected).abs() < 0.01);
        }

        #[test]
        fn prop_confidence_bounded(
            age in 18.0f32..80.0,
            income in 20000.0f32..200000.0,
            accounts in 0.0f32..20.0
        ) {
            let (model, _) = train_model().unwrap();
            let explainable = model.into_explainable();

            let sample = vec![age, income, accounts];
            let path = explainable.explain_one(&sample);

            let confidence = path.confidence();
            prop_assert!(confidence >= 0.0);
            prop_assert!(confidence <= 1.0);
        }

        #[test]
        fn prop_feature_contributions_count_matches(
            age in 18.0f32..80.0,
            income in 20000.0f32..200000.0,
            accounts in 0.0f32..20.0
        ) {
            let (model, _) = train_model().unwrap();
            let explainable = model.into_explainable();

            let sample = vec![age, income, accounts];
            let path = explainable.explain_one(&sample);

            // Should have one contribution per feature
            prop_assert_eq!(path.feature_contributions().len(), 3);
        }

        #[test]
        fn prop_deterministic_predictions(
            age in 18.0f32..80.0,
            income in 20000.0f32..200000.0,
            accounts in 0.0f32..20.0
        ) {
            let (model, _) = train_model().unwrap();
            let explainable = model.into_explainable();

            let sample = vec![age, income, accounts];

            let (outputs1, _) = explainable.predict_explained(&sample, 1);
            let (outputs2, _) = explainable.predict_explained(&sample, 1);

            prop_assert!((outputs1[0] - outputs2[0]).abs() < 1e-6);
        }
    }
}
