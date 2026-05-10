#![allow(clippy::needless_range_loop)]
//! Tier 1.3 tabular regression — shared helper for 5 recipes.
//!
//! Each recipe trains a linear regressor on a synthetic JSONL fixture
//! and asserts a closed-form falsifier (MSE→noise floor, MAE bound,
//! 1-step-ahead RMSE, etc.).
//!
//! All math is closed-form OLS via the normal equations on small
//! datasets — no SGD here. SGD-based recipes live in `sft_minimal`.

use crate::Result;

/// One row: a fixed-length feature vector + scalar target.
#[derive(Debug, Clone)]
pub struct Row {
    pub features: Vec<f64>,
    pub target: f64,
}

/// Fit a linear model y = w·x via OLS normal equations.
/// Returns the fitted weight vector and residual SS / N (MSE).
#[must_use]
pub fn fit_ols(rows: &[Row]) -> (Vec<f64>, f64) {
    if rows.is_empty() {
        return (Vec::new(), f64::NAN);
    }
    let n_features = rows[0].features.len();
    if n_features == 0 {
        return (Vec::new(), f64::NAN);
    }
    let n = rows.len();

    // Build X^T X (n_features × n_features) and X^T y (n_features × 1).
    let mut xtx = vec![vec![0.0_f64; n_features]; n_features];
    let mut xty = vec![0.0_f64; n_features];
    for row in rows {
        for i in 0..n_features {
            xty[i] += row.features[i] * row.target;
            for j in 0..n_features {
                xtx[i][j] += row.features[i] * row.features[j];
            }
        }
    }

    // Solve via Gaussian elimination with partial pivoting.
    let weights = solve_linear_system(&mut xtx, &mut xty);

    // Compute MSE = sum_i (y_i - w·x_i)^2 / N
    let mut sum_sq = 0.0_f64;
    for row in rows {
        let pred: f64 = row
            .features
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        let err = pred - row.target;
        sum_sq += err * err;
    }
    let mse = sum_sq / n as f64;
    (weights, mse)
}

/// Solve A·x = b via Gaussian elimination. Mutates A and b in place
/// and returns x. Returns zeros if A is singular.
fn solve_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        // Partial pivoting: find max |a[k][i]| for k in i..n
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        if a[i][i].abs() < 1e-12 {
            return vec![0.0; n]; // singular
        }
        for k in (i + 1)..n {
            let factor = a[k][i] / a[i][i];
            b[k] -= factor * b[i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    x
}

/// Mean absolute error.
#[must_use]
pub fn mae(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return f64::NAN;
    }
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / predictions.len() as f64
}

/// Pearson correlation coefficient between two vectors.
#[must_use]
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NAN;
    }
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a < 1e-12 || var_b < 1e-12 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Root mean square error.
#[must_use]
pub fn rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return f64::NAN;
    }
    let n = predictions.len() as f64;
    let sum_sq: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

/// Load a JSONL fixture of `{"x": [..], "y": ..}` rows.
pub fn load_rows(path: &str, n_features: usize) -> Result<Vec<Row>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| crate::CookbookError::invalid_format(format!("read {path}: {e}")))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let features = parse_array_field(line, "x").ok_or_else(|| {
            crate::CookbookError::invalid_format(format!("line {i}: missing/invalid x array"))
        })?;
        if features.len() != n_features {
            return Err(crate::CookbookError::invalid_format(format!(
                "line {i}: expected {n_features} features, got {}",
                features.len()
            )));
        }
        let target = parse_float_field(line, "y")
            .ok_or_else(|| crate::CookbookError::invalid_format(format!("line {i}: missing y")))?;
        out.push(Row { features, target });
    }
    Ok(out)
}

fn parse_float_field(line: &str, key: &str) -> Option<f64> {
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
    fn ols_recovers_linear_coefficients() {
        // y = 2*x1 + 3*x2 (no noise) — OLS must recover (2, 3).
        let rows: Vec<Row> = (0..50)
            .map(|i| {
                let x1 = i as f64 / 25.0;
                let x2 = (i as f64 - 25.0) / 12.5;
                Row {
                    features: vec![x1, x2],
                    target: 2.0 * x1 + 3.0 * x2,
                }
            })
            .collect();
        let (weights, mse) = fit_ols(&rows);
        assert!((weights[0] - 2.0).abs() < 1e-6, "w1={}", weights[0]);
        assert!((weights[1] - 3.0).abs() < 1e-6, "w2={}", weights[1]);
        assert!(mse < 1e-12, "MSE on noiseless = {mse}");
    }

    #[test]
    fn ols_mse_converges_to_noise_floor() {
        // y = 2*x1 + 3*x2 + ε  with var(ε) = σ²; OLS MSE → σ².
        let sigma_sq = 0.04_f64; // σ = 0.2
        let rows: Vec<Row> = (0..100)
            .map(|i| {
                let x1 = i as f64 / 50.0;
                let x2 = (i as f64 - 50.0) / 25.0;
                let noise = ((i * 13 % 37) as f64 / 100.0 - 0.18) * 0.2; // rough σ=0.2
                Row {
                    features: vec![x1, x2],
                    target: 2.0 * x1 + 3.0 * x2 + noise,
                }
            })
            .collect();
        let (_, mse) = fit_ols(&rows);
        assert!(
            mse < sigma_sq * 1.5,
            "MSE {mse} should be near σ² {sigma_sq}"
        );
    }

    #[test]
    fn mae_helper() {
        let p = vec![1.0, 2.0, 3.0];
        let t = vec![1.5, 2.5, 3.5];
        assert!((mae(&p, &t) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rmse_helper() {
        let p = vec![0.0; 4];
        let t = vec![1.0, 2.0, 2.0, 1.0];
        let r = rmse(&p, &t);
        // RMSE = sqrt((1+4+4+1)/4) = sqrt(2.5) ≈ 1.581
        assert!((r - (10.0_f64 / 4.0).sqrt()).abs() < 1e-12);
    }
}
