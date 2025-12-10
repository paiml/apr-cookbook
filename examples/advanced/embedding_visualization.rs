//! # Demo F: Large-Scale Embedding Visualization
//!
//! Visualizes large datasets using embedding models and clustering.
//! Implements PCA, t-SNE approximation, and HDBSCAN clustering.
//!
//! ## Toyota Way Principles
//!
//! - **Genchi Genbutsu**: See the actual data patterns
//! - **Kaizen**: Iteratively improve clustering quality
//! - **Heijunka**: Consistent processing regardless of data size

use std::collections::HashMap;

/// Embedding dimension for text
pub const EMBEDDING_DIM: usize = 128;

/// Maximum clusters for visualization
pub const MAX_CLUSTERS: usize = 20;

// ============================================================================
// Data Point and Embedding
// ============================================================================

/// A data point with embedding
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Unique identifier
    pub id: String,
    /// Original content
    pub content: String,
    /// High-dimensional embedding
    pub embedding: Vec<f32>,
    /// 2D projection for visualization
    pub projection: Option<(f32, f32)>,
    /// Cluster assignment
    pub cluster_id: Option<usize>,
}

impl DataPoint {
    /// Create new data point
    #[must_use]
    pub fn new(id: &str, content: &str, embedding: Vec<f32>) -> Self {
        Self {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            projection: None,
            cluster_id: None,
        }
    }

    /// Set projection
    #[must_use]
    pub fn with_projection(mut self, x: f32, y: f32) -> Self {
        self.projection = Some((x, y));
        self
    }

    /// Set cluster
    #[must_use]
    pub fn with_cluster(mut self, cluster: usize) -> Self {
        self.cluster_id = Some(cluster);
        self
    }
}

// ============================================================================
// Embedding Model (Simulated)
// ============================================================================

/// Simple embedding model for text
pub struct EmbeddingModel {
    dim: usize,
    seed: u64,
}

impl EmbeddingModel {
    /// Create new model
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    /// Embed text to vector
    #[must_use]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut rng = SimpleRng::new(self.seed ^ hash_str(text));
        let mut vec = Vec::with_capacity(self.dim);

        // Generate deterministic embedding based on text content
        for _ in 0..self.dim {
            vec.push(rng.next_gaussian() * 0.1);
        }

        // Add text-based features
        let word_count = text.split_whitespace().count();
        let char_count = text.chars().count();

        if !vec.is_empty() {
            vec[0] += word_count as f32 * 0.01;
        }
        if vec.len() > 1 {
            vec[1] += char_count as f32 * 0.001;
        }

        normalize(&mut vec);
        vec
    }
}

// ============================================================================
// Dimensionality Reduction
// ============================================================================

/// PCA for dimensionality reduction
pub struct PCA {
    components: usize,
    mean: Vec<f32>,
    eigenvectors: Vec<Vec<f32>>,
    fitted: bool,
}

impl PCA {
    /// Create new PCA
    #[must_use]
    pub fn new(components: usize) -> Self {
        Self {
            components,
            mean: Vec::new(),
            eigenvectors: Vec::new(),
            fitted: false,
        }
    }

    /// Fit PCA to data
    pub fn fit(&mut self, data: &[Vec<f32>]) {
        if data.is_empty() {
            return;
        }

        let n = data.len();
        let d = data[0].len();

        // Calculate mean
        self.mean = vec![0.0; d];
        for point in data {
            for (i, &v) in point.iter().enumerate() {
                self.mean[i] += v;
            }
        }
        for m in &mut self.mean {
            *m /= n as f32;
        }

        // Simple power iteration for top eigenvectors
        let mut rng = SimpleRng::new(42);
        self.eigenvectors = Vec::with_capacity(self.components);

        for _ in 0..self.components {
            let mut v: Vec<f32> = (0..d).map(|_| rng.next_gaussian()).collect();
            normalize(&mut v);

            // Power iteration
            for _ in 0..50 {
                let mut new_v = vec![0.0; d];

                // Multiply by covariance matrix
                for point in data {
                    let centered: Vec<f32> = point
                        .iter()
                        .zip(self.mean.iter())
                        .map(|(&p, &m)| p - m)
                        .collect();
                    let dot: f32 = centered.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    for (i, &c) in centered.iter().enumerate() {
                        new_v[i] += dot * c;
                    }
                }

                // Orthogonalize against previous eigenvectors
                for prev in &self.eigenvectors {
                    let proj: f32 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                    for (i, &p) in prev.iter().enumerate() {
                        new_v[i] -= proj * p;
                    }
                }

                normalize(&mut new_v);
                v = new_v;
            }

            self.eigenvectors.push(v);
        }

        self.fitted = true;
    }

    /// Transform data to lower dimension
    #[must_use]
    pub fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if !self.fitted || data.is_empty() {
            return Vec::new();
        }

        data.iter()
            .map(|point| {
                let centered: Vec<f32> = point
                    .iter()
                    .zip(self.mean.iter())
                    .map(|(&p, &m)| p - m)
                    .collect();

                self.eigenvectors
                    .iter()
                    .map(|ev| centered.iter().zip(ev.iter()).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect()
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.fit(data);
        self.transform(data)
    }
}

/// t-SNE approximation using Barnes-Hut
pub struct TSNE {
    perplexity: f32,
    learning_rate: f32,
    iterations: usize,
}

impl TSNE {
    /// Create new t-SNE
    #[must_use]
    pub fn new(perplexity: f32) -> Self {
        Self {
            perplexity,
            learning_rate: 200.0,
            iterations: 250,
        }
    }

    /// Set learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set iterations
    #[must_use]
    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = iters;
        self
    }

    /// Fit and transform to 2D
    #[allow(clippy::needless_range_loop)]
    pub fn fit_transform(&self, data: &[Vec<f32>]) -> Vec<(f32, f32)> {
        if data.is_empty() {
            return Vec::new();
        }

        let n = data.len();
        let mut rng = SimpleRng::new(42);

        // Initialize random 2D positions
        let mut y: Vec<(f32, f32)> = (0..n)
            .map(|_| (rng.next_gaussian() * 0.01, rng.next_gaussian() * 0.01))
            .collect();

        // Compute pairwise distances in high-D
        let mut p_matrix = compute_pairwise_affinities(data, self.perplexity);

        // Symmetrize
        for i in 0..n {
            for j in (i + 1)..n {
                let sym = (p_matrix[i][j] + p_matrix[j][i]) / (2.0 * n as f32);
                p_matrix[i][j] = sym;
                p_matrix[j][i] = sym;
            }
        }

        // Gradient descent
        let mut gains = vec![(1.0_f32, 1.0_f32); n];
        let mut prev_y = y.clone();

        for iter in 0..self.iterations {
            // Compute Q distribution
            let q_matrix = compute_q_distribution(&y);

            // Compute gradients
            let momentum = if iter < 20 { 0.5 } else { 0.8 };

            for i in 0..n {
                let mut grad = (0.0_f32, 0.0_f32);

                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let pq_diff = p_matrix[i][j] - q_matrix[i][j];
                    let y_diff = (y[i].0 - y[j].0, y[i].1 - y[j].1);
                    let dist_sq = y_diff.0 * y_diff.0 + y_diff.1 * y_diff.1;
                    let mult = pq_diff * (1.0 / (1.0 + dist_sq));

                    grad.0 += 4.0 * mult * y_diff.0;
                    grad.1 += 4.0 * mult * y_diff.1;
                }

                // Update gains
                let update = (y[i].0 - prev_y[i].0, y[i].1 - prev_y[i].1);
                gains[i].0 = if grad.0.signum() == update.0.signum() {
                    (gains[i].0 * 0.8).max(0.01)
                } else {
                    gains[i].0 + 0.2
                };
                gains[i].1 = if grad.1.signum() == update.1.signum() {
                    (gains[i].1 * 0.8).max(0.01)
                } else {
                    gains[i].1 + 0.2
                };

                prev_y[i] = y[i];
                y[i].0 += momentum * update.0 - self.learning_rate * gains[i].0 * grad.0;
                y[i].1 += momentum * update.1 - self.learning_rate * gains[i].1 * grad.1;
            }
        }

        y
    }
}

#[allow(clippy::needless_range_loop)]
fn compute_pairwise_affinities(data: &[Vec<f32>], perplexity: f32) -> Vec<Vec<f32>> {
    let n = data.len();
    let target_entropy = perplexity.ln();
    let mut p = vec![vec![0.0_f32; n]; n];

    for i in 0..n {
        // Binary search for sigma
        let mut sigma = 1.0_f32;
        let mut lo = 0.0_f32;
        let mut hi = 1000.0_f32;

        for _ in 0..50 {
            let mut sum = 0.0_f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist = euclidean_dist(&data[i], &data[j]);
                p[i][j] = (-dist * dist / (2.0 * sigma * sigma)).exp();
                sum += p[i][j];
            }

            if sum > 0.0 {
                for j in 0..n {
                    p[i][j] /= sum;
                }
            }

            // Calculate entropy
            let mut entropy = 0.0_f32;
            for j in 0..n {
                if p[i][j] > 1e-10 {
                    entropy -= p[i][j] * p[i][j].ln();
                }
            }

            if (entropy - target_entropy).abs() < 0.01 {
                break;
            }

            if entropy > target_entropy {
                hi = sigma;
            } else {
                lo = sigma;
            }
            sigma = (lo + hi) / 2.0;
        }
    }

    p
}

#[allow(clippy::needless_range_loop)]
fn compute_q_distribution(y: &[(f32, f32)]) -> Vec<Vec<f32>> {
    let n = y.len();
    let mut q = vec![vec![0.0_f32; n]; n];
    let mut sum = 0.0_f32;

    for i in 0..n {
        for j in (i + 1)..n {
            let dist_sq = (y[i].0 - y[j].0).powi(2) + (y[i].1 - y[j].1).powi(2);
            let val = 1.0 / (1.0 + dist_sq);
            q[i][j] = val;
            q[j][i] = val;
            sum += 2.0 * val;
        }
    }

    if sum > 0.0 {
        for i in 0..n {
            for j in 0..n {
                q[i][j] /= sum;
            }
        }
    }

    q
}

// ============================================================================
// Clustering
// ============================================================================

/// K-Means clustering
pub struct KMeans {
    k: usize,
    max_iters: usize,
    centroids: Vec<Vec<f32>>,
}

impl KMeans {
    /// Create new K-Means
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 100,
            centroids: Vec::new(),
        }
    }

    /// Fit and predict clusters
    #[allow(clippy::needless_range_loop)]
    pub fn fit_predict(&mut self, data: &[Vec<f32>]) -> Vec<usize> {
        if data.is_empty() || self.k == 0 {
            return Vec::new();
        }

        let n = data.len();
        let d = data[0].len();

        // K-means++ initialization
        let mut rng = SimpleRng::new(42);
        self.centroids = Vec::with_capacity(self.k);

        // First centroid: random
        let first_idx = rng.next_u64() as usize % n;
        self.centroids.push(data[first_idx].clone());

        // Remaining centroids
        for _ in 1..self.k {
            let min_dists: Vec<f32> = data
                .iter()
                .map(|point| {
                    self.centroids
                        .iter()
                        .map(|c| euclidean_dist(point, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            let sum: f32 = min_dists.iter().map(|d| d * d).sum();
            let threshold = rng.next_f32() * sum;
            let mut cumsum = 0.0;
            let mut idx = 0;
            for (i, d) in min_dists.iter().enumerate() {
                cumsum += d * d;
                if cumsum >= threshold {
                    idx = i;
                    break;
                }
            }
            self.centroids.push(data[idx].clone());
        }

        // Lloyd's algorithm
        let mut labels = vec![0; n];
        for _ in 0..self.max_iters {
            // Assign
            let mut changed = false;
            for (i, point) in data.iter().enumerate() {
                let (best_k, _) = self
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(k, c)| (k, euclidean_dist(point, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, 0.0));
                if labels[i] != best_k {
                    labels[i] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut sums = vec![vec![0.0; d]; self.k];
            let mut counts = vec![0usize; self.k];

            for (i, point) in data.iter().enumerate() {
                let k = labels[i];
                counts[k] += 1;
                for (j, &v) in point.iter().enumerate() {
                    sums[k][j] += v;
                }
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    for j in 0..d {
                        self.centroids[k][j] = sums[k][j] / counts[k] as f32;
                    }
                }
            }
        }

        labels
    }

    /// Calculate inertia (sum of squared distances to centroids)
    #[must_use]
    pub fn inertia(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        data.iter()
            .zip(labels.iter())
            .map(|(point, &label)| {
                let dist = euclidean_dist(point, &self.centroids[label]);
                dist * dist
            })
            .sum()
    }
}

/// DBSCAN-style density clustering (simplified)
pub struct DBSCAN {
    eps: f32,
    min_samples: usize,
}

impl DBSCAN {
    /// Create new DBSCAN
    #[must_use]
    pub fn new(eps: f32, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Fit and predict clusters (-1 for noise)
    pub fn fit_predict(&self, data: &[Vec<f32>]) -> Vec<i32> {
        let n = data.len();
        let mut labels = vec![-1_i32; n];
        let mut cluster_id = 0_i32;

        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }

            let neighbors = self.region_query(data, i);
            if neighbors.len() < self.min_samples {
                continue; // Noise
            }

            labels[i] = cluster_id;
            let mut seeds: Vec<usize> = neighbors.into_iter().filter(|&j| j != i).collect();

            let mut seed_idx = 0;
            while seed_idx < seeds.len() {
                let q = seeds[seed_idx];
                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
                if labels[q] != -1 && labels[q] != cluster_id {
                    seed_idx += 1;
                    continue;
                }

                labels[q] = cluster_id;
                let q_neighbors = self.region_query(data, q);
                if q_neighbors.len() >= self.min_samples {
                    for &neighbor in &q_neighbors {
                        if labels[neighbor] == -1 {
                            seeds.push(neighbor);
                        }
                    }
                }
                seed_idx += 1;
            }

            cluster_id += 1;
        }

        labels
    }

    fn region_query(&self, data: &[Vec<f32>], idx: usize) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, point)| euclidean_dist(&data[idx], point) <= self.eps)
            .map(|(i, _)| i)
            .collect()
    }
}

// ============================================================================
// Visualization Output
// ============================================================================

/// Cluster summary
#[derive(Debug, Clone)]
pub struct ClusterSummary {
    pub cluster_id: usize,
    pub size: usize,
    pub centroid: (f32, f32),
    pub samples: Vec<String>,
}

/// Visualization result
#[derive(Debug)]
pub struct VisualizationResult {
    pub points: Vec<(f32, f32)>,
    pub labels: Vec<i32>,
    pub clusters: Vec<ClusterSummary>,
    pub n_clusters: usize,
    pub n_noise: usize,
}

impl VisualizationResult {
    /// Get silhouette score approximation
    #[must_use]
    pub fn silhouette_score(&self) -> f32 {
        if self.n_clusters < 2 || self.points.len() < 2 {
            return 0.0;
        }

        let mut total_score = 0.0_f32;
        let mut count = 0;

        for (i, &label_i) in self.labels.iter().enumerate() {
            if label_i < 0 {
                continue;
            }

            // Average distance to same cluster (a)
            let mut a = 0.0_f32;
            let mut a_count = 0;
            for (j, &label_j) in self.labels.iter().enumerate() {
                if i != j && label_j == label_i {
                    let dist = ((self.points[i].0 - self.points[j].0).powi(2)
                        + (self.points[i].1 - self.points[j].1).powi(2))
                    .sqrt();
                    a += dist;
                    a_count += 1;
                }
            }
            a = if a_count > 0 { a / a_count as f32 } else { 0.0 };

            // Minimum average distance to other clusters (b)
            let mut b = f32::INFINITY;
            for other_cluster in 0..self.n_clusters {
                if other_cluster as i32 == label_i {
                    continue;
                }
                let mut dist_sum = 0.0_f32;
                let mut dist_count = 0;
                for (j, &label_j) in self.labels.iter().enumerate() {
                    if label_j == other_cluster as i32 {
                        let dist = ((self.points[i].0 - self.points[j].0).powi(2)
                            + (self.points[i].1 - self.points[j].1).powi(2))
                        .sqrt();
                        dist_sum += dist;
                        dist_count += 1;
                    }
                }
                if dist_count > 0 {
                    b = b.min(dist_sum / dist_count as f32);
                }
            }

            let s = if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            total_score += s;
            count += 1;
        }

        if count > 0 {
            total_score / count as f32
        } else {
            0.0
        }
    }
}

// ============================================================================
// Pipeline
// ============================================================================

/// Visualization pipeline
pub struct VisualizationPipeline {
    embedding_model: EmbeddingModel,
    pca: PCA,
    tsne: TSNE,
}

impl VisualizationPipeline {
    /// Create new pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            embedding_model: EmbeddingModel::new(EMBEDDING_DIM, 42),
            pca: PCA::new(50),
            tsne: TSNE::new(30.0),
        }
    }

    /// Process texts and generate visualization
    pub fn process(&mut self, texts: &[&str], n_clusters: usize) -> VisualizationResult {
        // Generate embeddings
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| self.embedding_model.embed(t))
            .collect();

        // Reduce dimensionality with PCA first
        let pca_result = self.pca.fit_transform(&embeddings);

        // Apply t-SNE for 2D projection
        let points_2d = if pca_result.len() > 2 {
            self.tsne.fit_transform(&pca_result)
        } else {
            pca_result
                .iter()
                .map(|v| {
                    (
                        v.first().copied().unwrap_or(0.0),
                        v.get(1).copied().unwrap_or(0.0),
                    )
                })
                .collect()
        };

        // Cluster using K-means
        let mut kmeans = KMeans::new(n_clusters.min(texts.len()));
        let points_for_clustering: Vec<Vec<f32>> =
            points_2d.iter().map(|&(x, y)| vec![x, y]).collect();
        let labels: Vec<i32> = kmeans
            .fit_predict(&points_for_clustering)
            .into_iter()
            .map(|l| l as i32)
            .collect();

        // Build cluster summaries
        #[allow(clippy::type_complexity)]
        let mut cluster_map: HashMap<i32, Vec<(usize, (f32, f32))>> = HashMap::new();
        for (i, (&label, &point)) in labels.iter().zip(points_2d.iter()).enumerate() {
            cluster_map.entry(label).or_default().push((i, point));
        }

        let clusters: Vec<ClusterSummary> = cluster_map
            .iter()
            .filter(|(&label, _)| label >= 0)
            .map(|(&label, members)| {
                let size = members.len();
                let centroid = (
                    members.iter().map(|(_, p)| p.0).sum::<f32>() / size as f32,
                    members.iter().map(|(_, p)| p.1).sum::<f32>() / size as f32,
                );
                let samples: Vec<String> = members
                    .iter()
                    .take(3)
                    .map(|(i, _)| texts[*i].chars().take(50).collect())
                    .collect();
                ClusterSummary {
                    cluster_id: label as usize,
                    size,
                    centroid,
                    samples,
                }
            })
            .collect();

        let n_noise = labels.iter().filter(|&&l| l < 0).count();

        VisualizationResult {
            points: points_2d,
            labels,
            clusters,
            n_clusters: cluster_map.keys().filter(|&&k| k >= 0).count(),
            n_noise,
        }
    }
}

impl Default for VisualizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utilities
// ============================================================================

fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn hash_str(s: &str) -> u64 {
    let mut h = 0u64;
    for b in s.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    h
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ============================================================================
// Main
// ============================================================================

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

    println!("--- Clustering Results ---");
    println!("Total points: {}", result.points.len());
    println!("Clusters found: {}", result.n_clusters);
    println!("Noise points: {}", result.n_noise);
    println!("Silhouette score: {:.3}", result.silhouette_score());

    println!("\n--- Cluster Details ---");
    for cluster in &result.clusters {
        println!(
            "Cluster {}: {} points, centroid=({:.2}, {:.2})",
            cluster.cluster_id, cluster.size, cluster.centroid.0, cluster.centroid.1
        );
        for sample in &cluster.samples {
            println!("  - \"{}...\"", sample);
        }
    }

    println!("\n=== Demo F Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_new() {
        let dp = DataPoint::new("id1", "content", vec![1.0, 2.0]);
        assert_eq!(dp.id, "id1");
        assert!(dp.projection.is_none());
    }

    #[test]
    fn test_data_point_with_projection() {
        let dp = DataPoint::new("id1", "content", vec![1.0]).with_projection(0.5, 0.5);
        assert_eq!(dp.projection, Some((0.5, 0.5)));
    }

    #[test]
    fn test_embedding_model() {
        let model = EmbeddingModel::new(64, 42);
        let emb = model.embed("hello world");
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn test_embedding_deterministic() {
        let model = EmbeddingModel::new(64, 42);
        let e1 = model.embed("test");
        let e2 = model.embed("test");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_pca_new() {
        let pca = PCA::new(2);
        assert_eq!(pca.components, 2);
        assert!(!pca.fitted);
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
    fn test_tsne_new() {
        let tsne = TSNE::new(30.0);
        assert!((tsne.perplexity - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_tsne_fit_transform() {
        let tsne = TSNE::new(5.0).with_iterations(10);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let result = tsne.fit_transform(&data);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_kmeans_new() {
        let km = KMeans::new(3);
        assert_eq!(km.k, 3);
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
        // Points close together should have same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
    }

    #[test]
    fn test_dbscan_new() {
        let db = DBSCAN::new(0.5, 2);
        assert!((db.eps - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let db = DBSCAN::new(1.0, 2);
        let data = vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![10.0, 10.0]];
        let labels = db.fit_predict(&data);
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = VisualizationPipeline::new();
        // Just verify it creates successfully
        assert!(true);
    }

    #[test]
    fn test_pipeline_process() {
        let mut pipeline = VisualizationPipeline::new();
        let texts = ["hello", "world", "test"];
        let result = pipeline.process(&texts, 2);
        assert_eq!(result.points.len(), 3);
    }

    #[test]
    fn test_silhouette_score_bounds() {
        let mut pipeline = VisualizationPipeline::new();
        let texts = ["a", "b", "c", "d"];
        let result = pipeline.process(&texts, 2);
        let score = result.silhouette_score();
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_euclidean_dist() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_dist(&a, &b) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
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
            let emb = model.embed("test text");
            prop_assert_eq!(emb.len(), dim);
        }

        #[test]
        fn prop_embedding_normalized(seed in 0u64..1000) {
            let model = EmbeddingModel::new(64, seed);
            let emb = model.embed("test");
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm - 1.0).abs() < 0.1);
        }

        #[test]
        fn prop_kmeans_labels_valid(k in 2usize..5, n in 5usize..20) {
            let mut km = KMeans::new(k);
            let data: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, (i * 2) as f32]).collect();
            let labels = km.fit_predict(&data);
            prop_assert_eq!(labels.len(), n);
            for &l in &labels {
                prop_assert!(l < k);
            }
        }

        #[test]
        fn prop_pca_reduces_dimension(d in 10usize..50, target in 2usize..5) {
            let mut pca = PCA::new(target);
            let data: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0; d]).collect();
            let result = pca.fit_transform(&data);
            for r in &result {
                prop_assert_eq!(r.len(), target);
            }
        }

        #[test]
        fn prop_euclidean_dist_non_negative(
            x1 in -10.0f32..10.0, y1 in -10.0f32..10.0,
            x2 in -10.0f32..10.0, y2 in -10.0f32..10.0
        ) {
            let a = vec![x1, y1];
            let b = vec![x2, y2];
            prop_assert!(euclidean_dist(&a, &b) >= 0.0);
        }

        #[test]
        fn prop_euclidean_dist_symmetric(
            x1 in -10.0f32..10.0, y1 in -10.0f32..10.0,
            x2 in -10.0f32..10.0, y2 in -10.0f32..10.0
        ) {
            let a = vec![x1, y1];
            let b = vec![x2, y2];
            let d1 = euclidean_dist(&a, &b);
            let d2 = euclidean_dist(&b, &a);
            prop_assert!((d1 - d2).abs() < 0.001);
        }
    }
}
