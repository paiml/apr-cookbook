#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::HashMap;

#[allow(clippy::needless_range_loop)]
pub fn compute_pairwise_affinities(data: &[Vec<f32>], perplexity: f32) -> Vec<Vec<f32>> {
    let n = data.len();
    let target_entropy = perplexity.ln();
    let mut p = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        let mut sigma = 1.0_f32;
        let (mut lo, mut hi) = (0.0_f32, 1000.0_f32);
        for _ in 0..50 {
            // Compute row probabilities
            let mut sum = 0.0_f32;
            for j in 0..n {
                if i == j {
                    p[i][j] = 0.0;
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
            let entropy: f32 =
                p[i].iter().fold(
                    0.0,
                    |acc, &v| if v > 1e-10 { acc - v * v.ln() } else { acc },
                );
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
pub fn compute_q_distribution(y: &[(f32, f32)]) -> Vec<Vec<f32>> {
    let n = y.len();
    let mut q = vec![vec![0.0_f32; n]; n];
    let mut sum = 0.0_f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let val = 1.0 / (1.0 + (y[i].0 - y[j].0).powi(2) + (y[i].1 - y[j].1).powi(2));
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

pub fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn hash_str(s: &str) -> u64 {
    s.bytes()
        .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)))
}
