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

// --- Visualization result types ---

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
