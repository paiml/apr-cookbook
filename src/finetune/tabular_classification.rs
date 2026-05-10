//! Tier 1.4 tabular classification — shared helpers.
//!
//! Each recipe trains a closed-form classifier on a synthetic JSONL
//! fixture and asserts a closed-form falsifier (perfect-acc on
//! linearly-separable, top-k acc threshold, minority-recall floor, etc.).
//!
//! For binary tasks: nearest-class-mean (NCM) — a deterministic linear
//! classifier that's exact on linearly separable data without any
//! optimization loop. For multi-class: per-class NCM with argmax.

use crate::Result;

/// One labeled tabular row.
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: u32,
}

/// Fit a nearest-class-mean classifier: for each class c, store μ_c.
/// Predict: argmin_c ||x - μ_c||₂².
#[must_use]
pub fn fit_ncm(samples: &[Sample], n_classes: u32) -> Vec<Vec<f64>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n_features = samples[0].features.len();
    let mut means = vec![vec![0.0_f64; n_features]; n_classes as usize];
    let mut counts = vec![0u32; n_classes as usize];
    for s in samples {
        let c = s.label as usize;
        if c >= means.len() {
            continue;
        }
        for (i, x) in s.features.iter().enumerate() {
            means[c][i] += x;
        }
        counts[c] += 1;
    }
    for (c, count) in counts.iter().enumerate() {
        if *count > 0 {
            for v in &mut means[c] {
                *v /= f64::from(*count);
            }
        }
    }
    means
}

/// Predict a class via NCM (argmin distance to class mean).
#[must_use]
pub fn predict_ncm(class_means: &[Vec<f64>], features: &[f64]) -> u32 {
    let mut best_class = 0u32;
    let mut best_dist = f64::INFINITY;
    for (c, mean) in class_means.iter().enumerate() {
        let mut sum_sq = 0.0_f64;
        for (i, m) in mean.iter().enumerate() {
            let d = features[i] - m;
            sum_sq += d * d;
        }
        if sum_sq < best_dist {
            best_dist = sum_sq;
            best_class = c as u32;
        }
    }
    best_class
}

/// Top-k accuracy: prediction is correct if the true label is among
/// the k smallest-distance classes.
#[must_use]
pub fn predict_topk(class_means: &[Vec<f64>], features: &[f64], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f64)> = class_means
        .iter()
        .enumerate()
        .map(|(c, mean)| {
            let sum_sq: f64 = mean
                .iter()
                .zip(features.iter())
                .map(|(m, x)| {
                    let d = x - m;
                    d * d
                })
                .sum();
            (c as u32, sum_sq)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.into_iter().take(k).map(|(c, _)| c).collect()
}

/// Per-class recall: for each class, the fraction of true-class samples
/// correctly classified. Returns one f64 per class.
#[must_use]
pub fn per_class_recall(predictions: &[u32], labels: &[u32], n_classes: u32) -> Vec<f64> {
    let mut tp = vec![0u32; n_classes as usize];
    let mut total = vec![0u32; n_classes as usize];
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        if l < n_classes {
            total[l as usize] += 1;
            if p == l {
                tp[l as usize] += 1;
            }
        }
    }
    tp.iter()
        .zip(total.iter())
        .map(|(t, n)| {
            if *n > 0 {
                f64::from(*t) / f64::from(*n)
            } else {
                0.0
            }
        })
        .collect()
}

/// Macro-F1 (mean of per-class F1s).
#[must_use]
pub fn macro_f1(predictions: &[u32], labels: &[u32], n_classes: u32) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0u32;
    for c in 0..n_classes {
        let tp = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p == c && **l == c)
            .count() as u32;
        let fp = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p == c && **l != c)
            .count() as u32;
        let fn_ = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| **p != c && **l == c)
            .count() as u32;
        let f1 = if tp == 0 {
            0.0
        } else {
            let p = f64::from(tp) / f64::from(tp + fp);
            let r = f64::from(tp) / f64::from(tp + fn_);
            if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            }
        };
        sum += f1;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

/// Load a JSONL fixture of `{"x": [..], "label": <int>}` rows.
pub fn load_samples(path: &str, n_features: usize) -> Result<Vec<Sample>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| crate::CookbookError::invalid_format(format!("read {path}: {e}")))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let features = parse_array_field(line, "x")
            .ok_or_else(|| crate::CookbookError::invalid_format(format!("line {i}: missing x")))?;
        if features.len() != n_features {
            return Err(crate::CookbookError::invalid_format(format!(
                "line {i}: expected {n_features} features, got {}",
                features.len()
            )));
        }
        let label = parse_int_field(line, "label").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing label"))
        })?;
        out.push(Sample { features, label });
    }
    Ok(out)
}

fn parse_int_field(line: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn parse_array_field(line: &str, key: &str) -> Option<Vec<f64>> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let rest = rest.strip_prefix('[')?;
    let end = rest.find(']')?;
    rest[..end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncm_perfect_on_linearly_separable() {
        // Two clearly-separated 2D Gaussians around (-1,-1) and (1,1).
        let mut samples = Vec::new();
        for i in 0..50 {
            samples.push(Sample {
                features: vec![-1.0 + (i as f64) * 0.001, -1.0 + (i as f64) * 0.001],
                label: 0,
            });
            samples.push(Sample {
                features: vec![1.0 + (i as f64) * 0.001, 1.0 + (i as f64) * 0.001],
                label: 1,
            });
        }
        let means = fit_ncm(&samples, 2);
        let preds: Vec<u32> = samples
            .iter()
            .map(|s| predict_ncm(&means, &s.features))
            .collect();
        let labels: Vec<u32> = samples.iter().map(|s| s.label).collect();
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| p == l)
            .count();
        assert_eq!(
            correct,
            samples.len(),
            "linearly-separable should be 100% correct"
        );
    }

    #[test]
    fn topk_includes_true_class_when_close() {
        let means = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        // Point near class 1 (1,0): top-2 should include class 1.
        let topk = predict_topk(&means, &[0.9, 0.1], 2);
        assert!(topk.contains(&1));
    }

    #[test]
    fn macro_f1_balanced_perfect() {
        let p = vec![0, 1, 2, 0, 1, 2];
        let l = p.clone();
        let f = macro_f1(&p, &l, 3);
        assert!((f - 1.0).abs() < 1e-12);
    }

    #[test]
    fn per_class_recall_majority_predictor_zero_minority() {
        // 9 majority (label=0), 1 minority (label=1); always predict 0.
        let labels: Vec<u32> = vec![0; 9].into_iter().chain(std::iter::once(1)).collect();
        let preds = vec![0u32; 10];
        let r = per_class_recall(&preds, &labels, 2);
        assert!((r[0] - 1.0).abs() < 1e-12);
        assert!(r[1].abs() < 1e-12);
    }
}
