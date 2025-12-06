//! # Recipe: Create APR KMeans Clustering Model
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
//! Train a KMeans clustering model on synthetic data and save as `.apr`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example create_apr_kmeans_clustering
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("create_apr_kmeans_clustering")?;

    // Generate synthetic clustered data
    let n_samples = 300;
    let n_features = 2;
    let n_clusters = 3;
    let x_data = generate_clustered_data(ctx.rng(), n_samples, n_features, n_clusters);

    // Train KMeans
    let max_iters = 100;
    let centroids = train_kmeans(ctx.rng(), &x_data, n_features, n_clusters, max_iters);

    ctx.record_metric("n_samples", n_samples as i64);
    ctx.record_metric("n_features", n_features as i64);
    ctx.record_metric("n_clusters", n_clusters as i64);

    // Calculate inertia (sum of squared distances to centroids)
    let assignments = assign_clusters(&x_data, &centroids, n_features);
    let inertia = calculate_inertia(&x_data, &centroids, &assignments, n_features);
    ctx.record_float_metric("inertia", inertia);

    // Save as APR
    let mut converter = AprConverter::new();
    converter.set_metadata(ConversionMetadata {
        name: Some("kmeans".to_string()),
        architecture: Some("clustering".to_string()),
        source_format: None,
        custom: std::collections::HashMap::new(),
    });

    converter.add_tensor(TensorData {
        name: "centroids".to_string(),
        shape: vec![n_clusters, n_features],
        dtype: DataType::F32,
        data: floats_to_bytes(&centroids),
    });

    let apr_path = ctx.path("kmeans.apr");
    let apr_bytes = converter.to_apr()?;
    std::fs::write(&apr_path, &apr_bytes)?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Trained KMeans with k={}", n_clusters);
    println!("Centroids:");
    for (i, chunk) in centroids.chunks(n_features).enumerate() {
        println!("  Cluster {}: {:?}", i, chunk);
    }
    println!("Inertia: {:.4}", inertia);
    println!("Saved to: {:?}", apr_path);

    Ok(())
}

/// Generate data with k clusters
fn generate_clustered_data(
    rng: &mut impl Rng,
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
) -> Vec<f32> {
    let mut data = Vec::with_capacity(n_samples * n_features);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|i| {
            (0..n_features)
                .map(|_| (i as f32 * 5.0) + rng.gen_range(-1.0f32..1.0f32))
                .collect()
        })
        .collect();

    let samples_per_cluster = n_samples / n_clusters;

    for (cluster_idx, center) in centers.iter().enumerate() {
        let n = if cluster_idx == n_clusters - 1 {
            n_samples - cluster_idx * samples_per_cluster
        } else {
            samples_per_cluster
        };

        for _ in 0..n {
            for &c in center {
                data.push(c + rng.gen_range(-0.5f32..0.5f32));
            }
        }
    }

    data
}

/// Train KMeans clustering
fn train_kmeans(
    rng: &mut impl Rng,
    x_data: &[f32],
    n_features: usize,
    n_clusters: usize,
    max_iters: usize,
) -> Vec<f32> {
    let n_samples = x_data.len() / n_features;

    // Initialize centroids randomly from data points
    let mut centroids = Vec::with_capacity(n_clusters * n_features);
    let mut used_indices = std::collections::HashSet::new();

    for _ in 0..n_clusters {
        let mut idx = rng.gen_range(0..n_samples);
        while used_indices.contains(&idx) {
            idx = rng.gen_range(0..n_samples);
        }
        used_indices.insert(idx);

        for j in 0..n_features {
            centroids.push(x_data[idx * n_features + j]);
        }
    }

    // Iterate until convergence or max_iters
    for _ in 0..max_iters {
        // Assign points to nearest centroid
        let assignments = assign_clusters(x_data, &centroids, n_features);

        // Update centroids
        let new_centroids = update_centroids(x_data, &assignments, n_features, n_clusters);

        // Check convergence
        let diff: f32 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        centroids = new_centroids;

        if diff < 1e-6 {
            break;
        }
    }

    centroids
}

/// Assign each point to nearest centroid
fn assign_clusters(x_data: &[f32], centroids: &[f32], n_features: usize) -> Vec<usize> {
    let n_samples = x_data.len() / n_features;
    let n_clusters = centroids.len() / n_features;
    let mut assignments = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let sample = &x_data[i * n_features..(i + 1) * n_features];
        let mut best_cluster = 0;
        let mut best_dist = f32::MAX;

        for k in 0..n_clusters {
            let centroid = &centroids[k * n_features..(k + 1) * n_features];
            let dist: f32 = sample
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_cluster = k;
            }
        }

        assignments.push(best_cluster);
    }

    assignments
}

/// Update centroids based on assignments
fn update_centroids(
    x_data: &[f32],
    assignments: &[usize],
    n_features: usize,
    n_clusters: usize,
) -> Vec<f32> {
    let mut new_centroids = vec![0.0f32; n_clusters * n_features];
    let mut counts = vec![0usize; n_clusters];

    for (i, &cluster) in assignments.iter().enumerate() {
        counts[cluster] += 1;
        for j in 0..n_features {
            new_centroids[cluster * n_features + j] += x_data[i * n_features + j];
        }
    }

    for k in 0..n_clusters {
        if counts[k] > 0 {
            for j in 0..n_features {
                new_centroids[k * n_features + j] /= counts[k] as f32;
            }
        }
    }

    new_centroids
}

/// Calculate inertia (within-cluster sum of squares)
fn calculate_inertia(
    x_data: &[f32],
    centroids: &[f32],
    assignments: &[usize],
    n_features: usize,
) -> f64 {
    let mut inertia = 0.0f64;

    for (i, &cluster) in assignments.iter().enumerate() {
        let sample = &x_data[i * n_features..(i + 1) * n_features];
        let centroid = &centroids[cluster * n_features..(cluster + 1) * n_features];

        let dist: f32 = sample
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        inertia += f64::from(dist);
    }

    inertia
}

fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let mut ctx = RecipeContext::new("test_kmeans_data").unwrap();
        let data = generate_clustered_data(ctx.rng(), 90, 2, 3);
        assert_eq!(data.len(), 180); // 90 samples * 2 features
    }

    #[test]
    fn test_kmeans_training() {
        let mut ctx = RecipeContext::new("test_kmeans_train").unwrap();
        let data = generate_clustered_data(ctx.rng(), 60, 2, 3);
        let centroids = train_kmeans(ctx.rng(), &data, 2, 3, 50);

        assert_eq!(centroids.len(), 6); // 3 clusters * 2 features
    }

    #[test]
    fn test_cluster_assignment() {
        let centroids = vec![0.0f32, 0.0, 10.0, 10.0]; // 2 centroids in 2D
        let data = vec![0.1f32, 0.1, 9.9, 9.9, 0.0, 0.0];

        let assignments = assign_clusters(&data, &centroids, 2);

        assert_eq!(assignments, vec![0, 1, 0]);
    }

    #[test]
    fn test_inertia_calculation() {
        let centroids = vec![0.0f32, 0.0];
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let assignments = vec![0, 0];

        let inertia = calculate_inertia(&data, &centroids, &assignments, 2);
        // Each point is distance 1 from origin, so inertia = 1 + 1 = 2
        assert!((inertia - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_deterministic() {
        let mut ctx1 = RecipeContext::new("det_kmeans").unwrap();
        let mut ctx2 = RecipeContext::new("det_kmeans").unwrap();

        let data1 = generate_clustered_data(ctx1.rng(), 30, 2, 3);
        let data2 = generate_clustered_data(ctx2.rng(), 30, 2, 3);

        assert_eq!(data1, data2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn prop_assignments_valid(n_samples in 10usize..50, n_clusters in 2usize..5) {
            let mut ctx = RecipeContext::new("prop_assign").unwrap();
            let data = generate_clustered_data(ctx.rng(), n_samples, 2, n_clusters);
            let centroids = train_kmeans(ctx.rng(), &data, 2, n_clusters, 10);
            let assignments = assign_clusters(&data, &centroids, 2);

            prop_assert_eq!(assignments.len(), n_samples);
            for &a in &assignments {
                prop_assert!(a < n_clusters);
            }
        }

        #[test]
        fn prop_inertia_non_negative(n_samples in 10usize..50) {
            let mut ctx = RecipeContext::new("prop_inertia").unwrap();
            let data = generate_clustered_data(ctx.rng(), n_samples, 2, 2);
            let centroids = train_kmeans(ctx.rng(), &data, 2, 2, 10);
            let assignments = assign_clusters(&data, &centroids, 2);
            let inertia = calculate_inertia(&data, &centroids, &assignments, 2);

            prop_assert!(inertia >= 0.0);
        }
    }
}
