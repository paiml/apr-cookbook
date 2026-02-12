//! Entrenar Model Evaluation Example
//!
//! Demonstrates the entrenar eval module for computing classification metrics,
//! confusion matrices, and multi-class metric reports.
//!
//! # Evaluation Features
//!
//! - **Confusion Matrix**: TP/FP/FN/TN per class with accuracy
//! - **MultiClassMetrics**: Per-class Precision, Recall, F1 with averaging
//! - **Classification Report**: sklearn-style formatted report string
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_eval_metrics
//! ```

use entrenar::eval::classification::{
    classification_report, confusion_matrix, Average, MultiClassMetrics,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate synthetic classification predictions
fn generate_predictions(
    n_samples: usize,
    n_classes: usize,
    accuracy: f32,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut y_true = Vec::with_capacity(n_samples);
    let mut y_pred = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let true_class = i % n_classes;
        y_true.push(true_class);

        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();
        let random_val = h as f32 / u64::MAX as f32;

        if random_val < accuracy {
            y_pred.push(true_class);
        } else {
            let mut hasher2 = DefaultHasher::new();
            (seed, "error", i).hash(&mut hasher2);
            let wrong =
                (true_class + 1 + (hasher2.finish() as usize % (n_classes - 1))) % n_classes;
            y_pred.push(wrong);
        }
    }

    (y_true, y_pred)
}

fn main() {
    println!("=== Entrenar Model Evaluation Example ===\n");

    let n_samples = 300;
    let n_classes = 4;
    let class_names = ["cat", "dog", "bird", "fish"];

    // =========================================================================
    // Section 1: Generate Predictions
    // =========================================================================
    println!("1. Synthetic Classification Dataset");
    println!("   ─────────────────────────────────────────");
    println!("   Samples:  {}", n_samples);
    println!("   Classes:  {} ({:?})", n_classes, class_names);

    let (y_true, y_pred_good) = generate_predictions(n_samples, n_classes, 0.85, 42);
    let (_, y_pred_poor) = generate_predictions(n_samples, n_classes, 0.50, 99);
    println!("   Model A: ~85% target accuracy");
    println!("   Model B: ~50% target accuracy");
    println!();

    // =========================================================================
    // Section 2: Confusion Matrix
    // =========================================================================
    println!("2. Confusion Matrix (Model A)");
    println!("   ─────────────────────────────────────────");

    // Note: confusion_matrix takes (y_pred, y_true)
    let cm = confusion_matrix(&y_pred_good, &y_true);
    println!("   Overall accuracy: {:.1}%", cm.accuracy() * 100.0);
    println!();

    // Print confusion matrix
    print!("   {:>8}", "pred→");
    for name in &class_names {
        print!("{:>8}", name);
    }
    println!();

    let matrix = cm.matrix();
    for (i, name) in class_names.iter().enumerate() {
        print!("   {:>8}", name);
        for j in 0..n_classes {
            print!("{:>8}", matrix[i][j]);
        }
        println!();
    }
    println!();

    // =========================================================================
    // Section 3: Per-Class Metrics (MultiClassMetrics)
    // =========================================================================
    println!("3. Per-Class Metrics (Model A)");
    println!("   ─────────────────────────────────────────");

    let metrics = MultiClassMetrics::from_predictions(&y_pred_good, &y_true);

    println!(
        "   {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Class", "Precision", "Recall", "F1", "Support"
    );
    println!("   {}", "─".repeat(50));

    for (i, name) in class_names.iter().enumerate() {
        println!(
            "   {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
            name, metrics.precision[i], metrics.recall[i], metrics.f1[i], metrics.support[i],
        );
    }
    println!("   {}", "─".repeat(50));
    println!(
        "   {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
        "macro",
        metrics.precision_avg(Average::Macro),
        metrics.recall_avg(Average::Macro),
        metrics.f1_avg(Average::Macro),
        n_samples,
    );
    println!(
        "   {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
        "weighted",
        metrics.precision_avg(Average::Weighted),
        metrics.recall_avg(Average::Weighted),
        metrics.f1_avg(Average::Weighted),
        n_samples,
    );
    println!();

    // =========================================================================
    // Section 4: Classification Report (sklearn-style)
    // =========================================================================
    println!("4. Classification Report (Model A, sklearn-style)");
    println!("   ─────────────────────────────────────────");

    let report = classification_report(&y_pred_good, &y_true);
    for line in report.lines() {
        println!("   {}", line);
    }
    println!();

    // =========================================================================
    // Section 5: Model Comparison
    // =========================================================================
    println!("5. Model Comparison Leaderboard");
    println!("   ─────────────────────────────────────────");

    let cm_good = confusion_matrix(&y_pred_good, &y_true);
    let cm_poor = confusion_matrix(&y_pred_poor, &y_true);
    let metrics_good = MultiClassMetrics::from_predictions(&y_pred_good, &y_true);
    let metrics_poor = MultiClassMetrics::from_predictions(&y_pred_poor, &y_true);

    println!(
        "   {:>12} {:>10} {:>12} {:>12}",
        "Model", "Accuracy", "F1(macro)", "F1(weighted)"
    );
    println!("   {}", "─".repeat(50));
    println!(
        "   {:>12} {:>10.1}% {:>12.3} {:>12.3}",
        "Model A",
        cm_good.accuracy() * 100.0,
        metrics_good.f1_avg(Average::Macro),
        metrics_good.f1_avg(Average::Weighted),
    );
    println!(
        "   {:>12} {:>10.1}% {:>12.3} {:>12.3}",
        "Model B",
        cm_poor.accuracy() * 100.0,
        metrics_poor.f1_avg(Average::Macro),
        metrics_poor.f1_avg(Average::Weighted),
    );

    let winner = if cm_good.accuracy() > cm_poor.accuracy() {
        "Model A"
    } else {
        "Model B"
    };
    println!("   Winner: {}", winner);
    println!();

    // =========================================================================
    // Section 6: Per-Class Accuracy Visualization
    // =========================================================================
    println!("6. Per-Class Performance (Model A vs B)");
    println!("   ─────────────────────────────────────────");

    for (i, name) in class_names.iter().enumerate() {
        let f1_good = metrics_good.f1[i];
        let f1_poor = metrics_poor.f1[i];
        let bar_good = "█".repeat((f1_good * 20.0) as usize);
        let bar_poor = "░".repeat((f1_poor * 20.0) as usize);
        println!("   {:>6} A: {:<20} F1={:.3}", name, bar_good, f1_good);
        println!("   {:>6} B: {:<20} F1={:.3}", "", bar_poor, f1_poor);
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_predictions_size() {
        let (y_true, y_pred) = generate_predictions(100, 4, 0.8, 42);
        assert_eq!(y_true.len(), 100);
        assert_eq!(y_pred.len(), 100);
    }

    #[test]
    fn test_generate_predictions_deterministic() {
        let (y1, p1) = generate_predictions(50, 3, 0.7, 42);
        let (y2, p2) = generate_predictions(50, 3, 0.7, 42);
        assert_eq!(y1, y2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        let cm = confusion_matrix(&y_pred, &y_true);
        assert!((cm.accuracy() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_confusion_matrix_imperfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 0, 2, 0, 1, 1]; // 4/6 correct
        let cm = confusion_matrix(&y_pred, &y_true);
        let expected = 4.0 / 6.0;
        assert!(
            (cm.accuracy() - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            cm.accuracy()
        );
    }

    #[test]
    fn test_multiclass_metrics_perfect() {
        let y_true = vec![0, 1, 0, 1, 0, 1];
        let y_pred = vec![0, 1, 0, 1, 0, 1];
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        assert!((metrics.f1_avg(Average::Macro) - 1.0).abs() < 1e-5);
        assert!((metrics.precision_avg(Average::Macro) - 1.0).abs() < 1e-5);
        assert!((metrics.recall_avg(Average::Macro) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_higher_accuracy_param_produces_more_correct() {
        let n = 1000;
        let (y_true_h, y_pred_h) = generate_predictions(n, 4, 0.95, 42);
        let (y_true_l, y_pred_l) = generate_predictions(n, 4, 0.40, 42);

        let correct_h: usize = y_true_h
            .iter()
            .zip(y_pred_h.iter())
            .filter(|(a, b)| a == b)
            .count();
        let correct_l: usize = y_true_l
            .iter()
            .zip(y_pred_l.iter())
            .filter(|(a, b)| a == b)
            .count();

        assert!(correct_h > correct_l);
    }

    #[test]
    fn test_classification_report_not_empty() {
        let (y_true, y_pred) = generate_predictions(100, 3, 0.75, 42);
        let report = classification_report(&y_pred, &y_true);
        assert!(!report.is_empty());
        assert!(report.contains("precision"));
    }

    #[test]
    fn test_f1_between_zero_and_one() {
        let (y_true, y_pred) = generate_predictions(200, 4, 0.6, 42);
        let metrics = MultiClassMetrics::from_predictions(&y_pred, &y_true);
        let f1 = metrics.f1_avg(Average::Macro);
        assert!(f1 >= 0.0);
        assert!(f1 <= 1.0);
    }
}
