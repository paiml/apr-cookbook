//! # Recipe: Create APR Decision Tree Model
//!
//! **Category**: Model Creation
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
//! Build a simple decision tree classifier and save as `.apr`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example create_apr_decision_tree
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("create_apr_decision_tree")?;

    // Generate binary classification data
    let n_samples = 500;
    let n_features = 4;
    let (x_data, y_data) = generate_classification_data(ctx.rng(), n_samples, n_features);

    // Build decision tree
    let max_depth = 5;
    let tree = build_decision_tree(&x_data, &y_data, n_features, max_depth);

    ctx.record_metric("n_samples", n_samples as i64);
    ctx.record_metric("n_features", n_features as i64);
    ctx.record_metric("max_depth", max_depth as i64);
    ctx.record_metric("n_nodes", tree.nodes.len() as i64);

    // Evaluate accuracy
    let predictions = predict_all(&tree, &x_data, n_features);
    let accuracy = calculate_accuracy(&predictions, &y_data);
    ctx.record_float_metric("accuracy", accuracy);

    // Serialize tree to bytes
    let tree_bytes = serialize_tree(&tree)?;

    // Save as APR
    let mut converter = AprConverter::new();
    converter.set_metadata(ConversionMetadata {
        name: Some("decision-tree".to_string()),
        architecture: Some("tree".to_string()),
        source_format: None,
        custom: std::collections::HashMap::new(),
    });

    converter.add_tensor(TensorData {
        name: "tree_structure".to_string(),
        shape: vec![tree_bytes.len()],
        dtype: DataType::U8,
        data: tree_bytes,
    });

    let apr_path = ctx.path("decision_tree.apr");
    let apr_bytes = converter.to_apr()?;
    std::fs::write(&apr_path, &apr_bytes)?;

    println!("=== Recipe: {} ===", ctx.name());
    println!(
        "Built tree with {} nodes (max_depth={})",
        tree.nodes.len(),
        max_depth
    );
    println!("Training accuracy: {:.2}%", accuracy * 100.0);
    println!("Saved to: {:?}", apr_path);

    Ok(())
}

/// Decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index for split (None if leaf)
    pub feature_idx: Option<usize>,
    /// Threshold for split
    pub threshold: f32,
    /// Left child index
    pub left: Option<usize>,
    /// Right child index
    pub right: Option<usize>,
    /// Prediction value (for leaves)
    pub prediction: u8,
}

/// Decision tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
}

/// Generate binary classification data (two clusters)
fn generate_classification_data(
    rng: &mut impl Rng,
    n_samples: usize,
    n_features: usize,
) -> (Vec<f32>, Vec<u8>) {
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label = u8::from(i >= n_samples / 2);

        // Class 0: centered around (-2, -2, ...)
        // Class 1: centered around (2, 2, ...)
        let center = if label == 0 { -2.0f32 } else { 2.0f32 };

        for _ in 0..n_features {
            let x = center + rng.gen_range(-1.0f32..1.0f32);
            x_data.push(x);
        }
        y_data.push(label);
    }

    (x_data, y_data)
}

/// Build a decision tree using recursive splitting
fn build_decision_tree(
    x_data: &[f32],
    y_data: &[u8],
    n_features: usize,
    max_depth: usize,
) -> DecisionTree {
    let n_samples = y_data.len();
    let indices: Vec<usize> = (0..n_samples).collect();

    let mut nodes = Vec::new();
    build_node(
        x_data, y_data, n_features, &indices, 0, max_depth, &mut nodes,
    );

    DecisionTree { nodes }
}

fn build_node(
    x_data: &[f32],
    y_data: &[u8],
    n_features: usize,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let node_idx = nodes.len();

    // Count class distribution
    let n_class_0 = indices.iter().filter(|&&i| y_data[i] == 0).count();
    let n_class_1 = indices.len() - n_class_0;
    let majority_class = u8::from(n_class_0 < n_class_1);

    // Check stopping conditions
    if depth >= max_depth || indices.len() <= 2 || n_class_0 == 0 || n_class_1 == 0 {
        nodes.push(TreeNode {
            feature_idx: None,
            threshold: 0.0,
            left: None,
            right: None,
            prediction: majority_class,
        });
        return node_idx;
    }

    // Find best split
    let (best_feature, best_threshold) = find_best_split(x_data, y_data, n_features, indices);

    // Split indices
    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| x_data[i * n_features + best_feature] <= best_threshold);

    if left_indices.is_empty() || right_indices.is_empty() {
        nodes.push(TreeNode {
            feature_idx: None,
            threshold: 0.0,
            left: None,
            right: None,
            prediction: majority_class,
        });
        return node_idx;
    }

    // Add placeholder node
    nodes.push(TreeNode {
        feature_idx: Some(best_feature),
        threshold: best_threshold,
        left: None,
        right: None,
        prediction: majority_class,
    });

    // Recursively build children
    let left_idx = build_node(
        x_data,
        y_data,
        n_features,
        &left_indices,
        depth + 1,
        max_depth,
        nodes,
    );
    let right_idx = build_node(
        x_data,
        y_data,
        n_features,
        &right_indices,
        depth + 1,
        max_depth,
        nodes,
    );

    // Update node with children
    nodes[node_idx].left = Some(left_idx);
    nodes[node_idx].right = Some(right_idx);

    node_idx
}

fn find_best_split(
    x_data: &[f32],
    y_data: &[u8],
    n_features: usize,
    indices: &[usize],
) -> (usize, f32) {
    let mut best_feature = 0;
    let mut best_threshold = 0.0f32;
    let mut best_gini = f32::MAX;

    for feature in 0..n_features {
        // Get unique values for this feature
        let mut values: Vec<f32> = indices
            .iter()
            .map(|&i| x_data[i * n_features + feature])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        for window in values.windows(2) {
            let threshold = (window[0] + window[1]) / 2.0;
            let gini =
                calculate_split_gini(x_data, y_data, n_features, indices, feature, threshold);

            if gini < best_gini {
                best_gini = gini;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    (best_feature, best_threshold)
}

fn calculate_split_gini(
    x_data: &[f32],
    y_data: &[u8],
    n_features: usize,
    indices: &[usize],
    feature: usize,
    threshold: f32,
) -> f32 {
    let mut left_0 = 0usize;
    let mut left_1 = 0usize;
    let mut right_0 = 0usize;
    let mut right_1 = 0usize;

    for &i in indices {
        let x = x_data[i * n_features + feature];
        let y = y_data[i];

        if x <= threshold {
            if y == 0 {
                left_0 += 1;
            } else {
                left_1 += 1;
            }
        } else if y == 0 {
            right_0 += 1;
        } else {
            right_1 += 1;
        }
    }

    let left_total = left_0 + left_1;
    let right_total = right_0 + right_1;
    let total = left_total + right_total;

    if left_total == 0 || right_total == 0 {
        return f32::MAX;
    }

    let left_gini = 1.0
        - (left_0 as f32 / left_total as f32).powi(2)
        - (left_1 as f32 / left_total as f32).powi(2);
    let right_gini = 1.0
        - (right_0 as f32 / right_total as f32).powi(2)
        - (right_1 as f32 / right_total as f32).powi(2);

    (left_total as f32 * left_gini + right_total as f32 * right_gini) / total as f32
}

fn predict_all(tree: &DecisionTree, x_data: &[f32], n_features: usize) -> Vec<u8> {
    let n_samples = x_data.len() / n_features;
    let mut predictions = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let sample = &x_data[i * n_features..(i + 1) * n_features];
        predictions.push(predict_one(tree, sample));
    }

    predictions
}

fn predict_one(tree: &DecisionTree, sample: &[f32]) -> u8 {
    let mut node_idx = 0;

    loop {
        let node = &tree.nodes[node_idx];

        match node.feature_idx {
            None => return node.prediction,
            Some(feature) => {
                if sample[feature] <= node.threshold {
                    node_idx = node.left.unwrap_or(node_idx);
                } else {
                    node_idx = node.right.unwrap_or(node_idx);
                }
            }
        }

        // Safety check to prevent infinite loops
        if node_idx >= tree.nodes.len() {
            return tree.nodes[0].prediction;
        }
    }
}

fn calculate_accuracy(predictions: &[u8], targets: &[u8]) -> f64 {
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / predictions.len() as f64
}

fn serialize_tree(tree: &DecisionTree) -> Result<Vec<u8>> {
    serde_json::to_vec(tree).map_err(|e| CookbookError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let mut ctx = RecipeContext::new("test_tree_data").unwrap();
        let (x, y) = generate_classification_data(ctx.rng(), 100, 4);

        assert_eq!(x.len(), 400);
        assert_eq!(y.len(), 100);

        // Should have both classes
        let n_class_0 = y.iter().filter(|&&l| l == 0).count();
        let n_class_1 = y.iter().filter(|&&l| l == 1).count();
        assert_eq!(n_class_0, 50);
        assert_eq!(n_class_1, 50);
    }

    #[test]
    fn test_tree_building() {
        let mut ctx = RecipeContext::new("test_tree_build").unwrap();
        let (x, y) = generate_classification_data(ctx.rng(), 100, 2);
        let tree = build_decision_tree(&x, &y, 2, 3);

        assert!(!tree.nodes.is_empty());
        assert!(tree.nodes.len() <= 15); // Max 2^4 - 1 nodes for depth 3
    }

    #[test]
    fn test_prediction() {
        let mut ctx = RecipeContext::new("test_tree_predict").unwrap();
        let (x, y) = generate_classification_data(ctx.rng(), 200, 2);
        let tree = build_decision_tree(&x, &y, 2, 5);

        let predictions = predict_all(&tree, &x, 2);
        let accuracy = calculate_accuracy(&predictions, &y);

        // Should achieve reasonable accuracy on training data
        assert!(accuracy > 0.7, "Accuracy should be > 70%, got {}", accuracy);
    }

    #[test]
    fn test_serialization() {
        let tree = DecisionTree {
            nodes: vec![TreeNode {
                feature_idx: Some(0),
                threshold: 0.5,
                left: Some(1),
                right: Some(2),
                prediction: 0,
            }],
        };

        let bytes = serialize_tree(&tree).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_deterministic() {
        let mut ctx1 = RecipeContext::new("det_tree").unwrap();
        let mut ctx2 = RecipeContext::new("det_tree").unwrap();

        let (x1, y1) = generate_classification_data(ctx1.rng(), 50, 2);
        let (x2, y2) = generate_classification_data(ctx2.rng(), 50, 2);

        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_accuracy_bounded(n_samples in 10usize..100) {
            let mut ctx = RecipeContext::new("prop_accuracy").unwrap();
            let (x, y) = generate_classification_data(ctx.rng(), n_samples, 2);
            let tree = build_decision_tree(&x, &y, 2, 3);
            let predictions = predict_all(&tree, &x, 2);
            let accuracy = calculate_accuracy(&predictions, &y);

            prop_assert!(accuracy >= 0.0 && accuracy <= 1.0);
        }

        #[test]
        fn prop_tree_has_nodes(n_samples in 10usize..100, n_features in 1usize..5) {
            let mut ctx = RecipeContext::new("prop_tree_nodes").unwrap();
            let (x, y) = generate_classification_data(ctx.rng(), n_samples, n_features);
            let tree = build_decision_tree(&x, &y, n_features, 3);

            prop_assert!(!tree.nodes.is_empty());
        }
    }
}
