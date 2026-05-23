#![allow(unused_imports)]
//! Large-Scale Embedding Visualization
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! PCA, t-SNE approximation, K-Means and DBSCAN clustering for embeddings.
//!
//! ```bash
//! cargo run --example embedding_visualization
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
//! - McInnes, L. et al. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv:1802.03426

use std::collections::HashMap;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo F: Large-Scale Embedding Visualization ===\n");
    let texts = [
        "Machine learning is a branch of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing handles text data",
        "Computer vision processes images and video",
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "Rust is a systems programming language",
        "Python is popular for data science",
        "JavaScript runs in web browsers",
        "Go is designed for concurrent programming",
    ];
    let mut pipeline = VisualizationPipeline::new();
    let result = pipeline.process(&texts, 3);
    println!(
        "Points: {}, Clusters: {}, Noise: {}, Silhouette: {:.3}",
        result.points.len(),
        result.n_clusters,
        result.n_noise,
        result.silhouette_score()
    );
    for c in &result.clusters {
        println!(
            "  Cluster {}: {} points at ({:.2}, {:.2})",
            c.cluster_id, c.size, c.centroid.0, c.centroid.1
        );
        for s in &c.samples {
            println!("    - \"{s}...\"");
        }
    }
    println!("\n=== Demo F Complete ===");
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports, clippy::wildcard_imports)]
    use super::helpers::*;
    use super::*;

    #[test]
    fn test_embedding_model_and_determinism() {
        let model = EmbeddingModel::new(64, 42);
        let e1 = model.embed("hello world");
        let e2 = model.embed("hello world");
        assert_eq!(e1.len(), 64);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_pca_fit_transform() {
        let mut pca = PCA::new(2);
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = pca.fit_transform(&data);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_tsne_fit_transform() {
        let tsne = TSNE::new(5.0).with_iterations(10);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(tsne.fit_transform(&data).len(), 3);
    }

    #[test]
    fn test_kmeans_fit_predict() {
        let mut km = KMeans::new(2);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let labels = km.fit_predict(&data);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let db = DBSCAN::new(1.0, 2);
        let data = vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![10.0, 10.0]];
        assert_eq!(db.fit_predict(&data).len(), 3);
    }

    #[test]
    fn test_pipeline_process() {
        let mut pipeline = VisualizationPipeline::new();
        let result = pipeline.process(&["hello", "world", "test"], 2);
        assert_eq!(result.points.len(), 3);
    }

    #[test]
    fn test_silhouette_score_bounds() {
        let mut pipeline = VisualizationPipeline::new();
        let result = pipeline.process(&["a", "b", "c", "d"], 2);
        let score = result.silhouette_score();
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_euclidean_dist_and_normalize() {
        assert!((euclidean_dist(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 0.01);
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 0.01);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_embedding_length(dim in 16usize..256, seed in 0u64..1000) {
            let model = EmbeddingModel::new(dim, seed);
            prop_assert_eq!(model.embed("test text").len(), dim);
        }

        #[test]
        fn prop_kmeans_labels_valid(k in 2usize..5, n in 5usize..20) {
            let mut km = KMeans::new(k);
            let data: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, (i * 2) as f32]).collect();
            let labels = km.fit_predict(&data);
            prop_assert_eq!(labels.len(), n);
            for &l in &labels { prop_assert!(l < k); }
        }

        #[test]
        fn prop_euclidean_dist_non_negative(x1 in -10.0f32..10.0, y1 in -10.0f32..10.0, x2 in -10.0f32..10.0, y2 in -10.0f32..10.0) {
            prop_assert!(euclidean_dist(&[x1, y1], &[x2, y2]) >= 0.0);
        }
    }
}
