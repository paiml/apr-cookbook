#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports,
    clippy::upper_case_acronyms
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::helpers::*;
use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::HashMap;

pub const EMBEDDING_DIM: usize = 128;
pub const MAX_CLUSTERS: usize = 20;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    projection: Option<(f32, f32)>,
    pub cluster_id: Option<usize>,
}

impl DataPoint {
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
    #[must_use]
    pub fn with_projection(mut self, x: f32, y: f32) -> Self {
        self.projection = Some((x, y));
        self
    }
    #[must_use]
    pub fn with_cluster(mut self, cluster: usize) -> Self {
        self.cluster_id = Some(cluster);
        self
    }
}

pub struct EmbeddingModel {
    pub dim: usize,
    pub seed: u64,
}

impl EmbeddingModel {
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    #[must_use]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut rng = SimpleRng::new(self.seed ^ hash_str(text));
        let mut vec: Vec<f32> = (0..self.dim).map(|_| rng.next_gaussian() * 0.1).collect();
        if !vec.is_empty() {
            vec[0] += text.split_whitespace().count() as f32 * 0.01;
        }
        if vec.len() > 1 {
            vec[1] += text.chars().count() as f32 * 0.001;
        }
        normalize(&mut vec);
        vec
    }
}

// --- PCA ---

pub struct PCA {
    pub components: usize,
    pub mean: Vec<f32>,
    pub eigenvectors: Vec<Vec<f32>>,
    pub fitted: bool,
}

impl PCA {
    #[must_use]
    pub fn new(components: usize) -> Self {
        Self {
            components,
            mean: Vec::new(),
            eigenvectors: Vec::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) {
        if data.is_empty() {
            return;
        }
        let (n, d) = (data.len(), data[0].len());
        self.mean = vec![0.0; d];
        for point in data {
            for (i, &v) in point.iter().enumerate() {
                self.mean[i] += v;
            }
        }
        for m in &mut self.mean {
            *m /= n as f32;
        }
        let mut rng = SimpleRng::new(42);
        self.eigenvectors = Vec::with_capacity(self.components);
        for _ in 0..self.components {
            let mut v: Vec<f32> = (0..d).map(|_| rng.next_gaussian()).collect();
            normalize(&mut v);
            for _ in 0..50 {
                let mut new_v = vec![0.0; d];
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

    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.fit(data);
        self.transform(data)
    }
}

// --- t-SNE ---

pub struct TSNE {
    pub perplexity: f32,
    pub learning_rate: f32,
    pub iterations: usize,
}

impl TSNE {
    #[must_use]
    pub fn new(perplexity: f32) -> Self {
        Self {
            perplexity,
            learning_rate: 200.0,
            iterations: 250,
        }
    }
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
    #[must_use]
    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = iters;
        self
    }

    #[allow(clippy::needless_range_loop)]
    pub fn fit_transform(&self, data: &[Vec<f32>]) -> Vec<(f32, f32)> {
        if data.is_empty() {
            return Vec::new();
        }
        let n = data.len();
        let mut rng = SimpleRng::new(42);
        let mut y: Vec<(f32, f32)> = (0..n)
            .map(|_| (rng.next_gaussian() * 0.01, rng.next_gaussian() * 0.01))
            .collect();
        let mut p_matrix = compute_pairwise_affinities(data, self.perplexity);
        for i in 0..n {
            for j in (i + 1)..n {
                let sym = (p_matrix[i][j] + p_matrix[j][i]) / (2.0 * n as f32);
                p_matrix[i][j] = sym;
                p_matrix[j][i] = sym;
            }
        }
        let mut gains = vec![(1.0_f32, 1.0_f32); n];
        let mut prev_y = y.clone();
        for iter in 0..self.iterations {
            let q_matrix = compute_q_distribution(&y);
            let momentum = if iter < 20 { 0.5 } else { 0.8 };
            for i in 0..n {
                let mut grad = (0.0_f32, 0.0_f32);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let pq = p_matrix[i][j] - q_matrix[i][j];
                    let yd = (y[i].0 - y[j].0, y[i].1 - y[j].1);
                    let mult = pq / (1.0 + yd.0 * yd.0 + yd.1 * yd.1);
                    grad.0 += 4.0 * mult * yd.0;
                    grad.1 += 4.0 * mult * yd.1;
                }
                let upd = (y[i].0 - prev_y[i].0, y[i].1 - prev_y[i].1);
                gains[i].0 = if grad.0.signum() == upd.0.signum() {
                    (gains[i].0 * 0.8).max(0.01)
                } else {
                    gains[i].0 + 0.2
                };
                gains[i].1 = if grad.1.signum() == upd.1.signum() {
                    (gains[i].1 * 0.8).max(0.01)
                } else {
                    gains[i].1 + 0.2
                };
                prev_y[i] = y[i];
                y[i].0 += momentum * upd.0 - self.learning_rate * gains[i].0 * grad.0;
                y[i].1 += momentum * upd.1 - self.learning_rate * gains[i].1 * grad.1;
            }
        }
        y
    }
}

// --- K-Means ---

pub struct KMeans {
    pub k: usize,
    pub max_iters: usize,
    pub centroids: Vec<Vec<f32>>,
}

impl KMeans {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 100,
            centroids: Vec::new(),
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn fit_predict(&mut self, data: &[Vec<f32>]) -> Vec<usize> {
        if data.is_empty() || self.k == 0 {
            return Vec::new();
        }
        let (n, d) = (data.len(), data[0].len());
        let mut rng = SimpleRng::new(42);
        self.centroids = vec![data[rng.next_u64() as usize % n].clone()];
        for _ in 1..self.k {
            let dists: Vec<f32> = data
                .iter()
                .map(|p| {
                    self.centroids
                        .iter()
                        .map(|c| euclidean_dist(p, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();
            let sum: f32 = dists.iter().map(|d| d * d).sum();
            let thresh = rng.next_f32() * sum;
            let mut cum = 0.0;
            let mut idx = 0;
            for (i, d) in dists.iter().enumerate() {
                cum += d * d;
                if cum >= thresh {
                    idx = i;
                    break;
                }
            }
            self.centroids.push(data[idx].clone());
        }
        let mut labels = vec![0; n];
        for _ in 0..self.max_iters {
            let mut changed = false;
            for (i, point) in data.iter().enumerate() {
                let best = self
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(k, c)| (k, euclidean_dist(point, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map_or(0, |(k, _)| k);
                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            let mut sums = vec![vec![0.0; d]; self.k];
            let mut counts = vec![0usize; self.k];
            for (i, point) in data.iter().enumerate() {
                counts[labels[i]] += 1;
                for (j, &v) in point.iter().enumerate() {
                    sums[labels[i]][j] += v;
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

    #[must_use]
    pub fn inertia(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        data.iter()
            .zip(labels.iter())
            .map(|(p, &l)| {
                let d = euclidean_dist(p, &self.centroids[l]);
                d * d
            })
            .sum()
    }
}

// --- DBSCAN ---

pub struct DBSCAN {
    pub eps: f32,
    pub min_samples: usize,
}

impl DBSCAN {
    #[must_use]
    pub fn new(eps: f32, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    pub fn fit_predict(&self, data: &[Vec<f32>]) -> Vec<i32> {
        let n = data.len();
        let mut labels = vec![-1_i32; n];
        let mut cluster_id = 0_i32;
        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }
            let neighbors: Vec<usize> = data
                .iter()
                .enumerate()
                .filter(|(_, p)| euclidean_dist(&data[i], p) <= self.eps)
                .map(|(j, _)| j)
                .collect();
            if neighbors.len() < self.min_samples {
                continue;
            }
            labels[i] = cluster_id;
            let mut seeds: Vec<usize> = neighbors.into_iter().filter(|&j| j != i).collect();
            let mut si = 0;
            while si < seeds.len() {
                let q = seeds[si];
                if labels[q] != -1 && labels[q] != cluster_id {
                    si += 1;
                    continue;
                }
                labels[q] = cluster_id;
                let qn: Vec<usize> = data
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| euclidean_dist(&data[q], p) <= self.eps)
                    .map(|(j, _)| j)
                    .collect();
                if qn.len() >= self.min_samples {
                    for &nb in &qn {
                        if labels[nb] == -1 {
                            seeds.push(nb);
                        }
                    }
                }
                si += 1;
            }
            cluster_id += 1;
        }
        labels
    }
}

// --- Visualization ---

#[derive(Debug, Clone)]
pub struct ClusterSummary {
    pub cluster_id: usize,
    pub size: usize,
    pub centroid: (f32, f32),
    pub samples: Vec<String>,
}

#[derive(Debug)]
pub struct VisualizationResult {
    pub points: Vec<(f32, f32)>,
    pub labels: Vec<i32>,
    pub clusters: Vec<ClusterSummary>,
    pub n_clusters: usize,
    pub n_noise: usize,
}

impl VisualizationResult {
    #[must_use]
    pub fn silhouette_score(&self) -> f32 {
        if self.n_clusters < 2 || self.points.len() < 2 {
            return 0.0;
        }
        let (mut total, mut count) = (0.0_f32, 0);
        for (i, &li) in self.labels.iter().enumerate() {
            if li < 0 {
                continue;
            }
            let mut a = 0.0_f32;
            let mut ac = 0;
            for (j, &lj) in self.labels.iter().enumerate() {
                if i != j && lj == li {
                    a += ((self.points[i].0 - self.points[j].0).powi(2)
                        + (self.points[i].1 - self.points[j].1).powi(2))
                    .sqrt();
                    ac += 1;
                }
            }
            a = if ac > 0 { a / ac as f32 } else { 0.0 };
            let mut b = f32::INFINITY;
            for oc in 0..self.n_clusters {
                if oc as i32 == li {
                    continue;
                }
                let (mut ds, mut dc) = (0.0_f32, 0);
                for (j, &lj) in self.labels.iter().enumerate() {
                    if lj == oc as i32 {
                        ds += ((self.points[i].0 - self.points[j].0).powi(2)
                            + (self.points[i].1 - self.points[j].1).powi(2))
                        .sqrt();
                        dc += 1;
                    }
                }
                if dc > 0 {
                    b = b.min(ds / dc as f32);
                }
            }
            total += if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            count += 1;
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

// --- Pipeline ---

pub struct VisualizationPipeline {
    pub embedding_model: EmbeddingModel,
    pub pca: PCA,
    pub tsne: TSNE,
}

impl VisualizationPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self {
            embedding_model: EmbeddingModel::new(EMBEDDING_DIM, 42),
            pca: PCA::new(50),
            tsne: TSNE::new(30.0),
        }
    }

    pub fn process(&mut self, texts: &[&str], n_clusters: usize) -> VisualizationResult {
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| self.embedding_model.embed(t))
            .collect();
        let pca_result = self.pca.fit_transform(&embeddings);
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
        let mut kmeans = KMeans::new(n_clusters.min(texts.len()));
        let pts: Vec<Vec<f32>> = points_2d.iter().map(|&(x, y)| vec![x, y]).collect();
        let labels: Vec<i32> = kmeans
            .fit_predict(&pts)
            .into_iter()
            .map(|l| l as i32)
            .collect();
        #[allow(clippy::type_complexity)]
        let mut cmap: HashMap<i32, Vec<(usize, (f32, f32))>> = HashMap::new();
        for (i, (&l, &p)) in labels.iter().zip(points_2d.iter()).enumerate() {
            cmap.entry(l).or_default().push((i, p));
        }
        let clusters: Vec<ClusterSummary> = cmap
            .iter()
            .filter(|(&l, _)| l >= 0)
            .map(|(&l, members)| {
                let sz = members.len();
                ClusterSummary {
                    cluster_id: l as usize,
                    size: sz,
                    centroid: (
                        members.iter().map(|(_, p)| p.0).sum::<f32>() / sz as f32,
                        members.iter().map(|(_, p)| p.1).sum::<f32>() / sz as f32,
                    ),
                    samples: members
                        .iter()
                        .take(3)
                        .map(|(i, _)| texts[*i].chars().take(50).collect())
                        .collect(),
                }
            })
            .collect();
        let n_noise = labels.iter().filter(|&&l| l < 0).count();
        VisualizationResult {
            points: points_2d,
            labels,
            clusters,
            n_clusters: cmap.keys().filter(|&&k| k >= 0).count(),
            n_noise,
        }
    }
}

impl Default for VisualizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// --- Utilities ---

pub struct SimpleRng {
    pub state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
