//! Local explainability wrappers for inference monitoring
//!
//! These wrappers bridge aprender models with entrenar's inference monitoring system.
//! Provided locally to avoid the cyclic dependency between aprender and entrenar
//! when both are at the same version level (aprender ↔ entrenar cycle via
//! the `inference-monitoring` feature).

use aprender::linear_model::LinearRegression;
use entrenar::monitor::inference::{path::LinearPath, Explainable};

/// Wrapper that makes `LinearRegression` explainable for inference monitoring.
#[derive(Debug, Clone)]
pub struct LinearExplainable {
    model: LinearRegression,
}

impl LinearExplainable {
    pub fn new(model: LinearRegression) -> Self {
        let _ = model.coefficients();
        Self { model }
    }

    pub fn n_features(&self) -> usize {
        self.model.coefficients().len()
    }

    fn compute_contributions(&self, sample: &[f32]) -> Vec<f32> {
        let coefficients = self.model.coefficients();
        coefficients
            .as_slice()
            .iter()
            .zip(sample)
            .map(|(&w, &x)| w * x)
            .collect()
    }
}

impl Explainable for LinearExplainable {
    type Path = LinearPath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let n_features = self.n_features();
        assert_eq!(
            x.len(),
            n_features * n_samples,
            "Input length {} must equal n_features ({}) * n_samples ({})",
            x.len(),
            n_features,
            n_samples
        );

        let intercept = self.model.intercept();
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * n_features;
            let end = start + n_features;
            let sample = &x[start..end];

            let contributions = self.compute_contributions(sample);
            let logit: f32 = contributions.iter().sum::<f32>() + intercept;
            let output = logit;

            let path = LinearPath::new(contributions, intercept, logit, output);

            outputs.push(output);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let (_, paths) = self.predict_explained(sample, 1);
        paths.into_iter().next().expect("Should have one path")
    }
}

/// Extension trait to convert `LinearRegression` to explainable.
pub trait IntoExplainable {
    fn into_explainable(self) -> LinearExplainable;
}

impl IntoExplainable for LinearRegression {
    fn into_explainable(self) -> LinearExplainable {
        LinearExplainable::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::linear_model::LinearRegression;
    use aprender::Estimator;
    use entrenar::monitor::inference::Explainable;

    fn make_model() -> LinearRegression {
        // Use linearly independent rows to ensure positive definite X^T X
        let x = aprender::Matrix::from_vec(4, 2, vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0])
            .expect("valid matrix");
        let y = aprender::Vector::from_vec(vec![1.0f32, 2.0, 3.0, 8.0]);
        let mut model = LinearRegression::new();
        model.fit(&x, &y).expect("fit succeeded");
        model
    }

    #[test]
    fn test_new() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        assert_eq!(explainable.n_features(), 2);
    }

    #[test]
    fn test_into_explainable() {
        let model = make_model();
        let explainable = model.into_explainable();
        assert_eq!(explainable.n_features(), 2);
    }

    #[test]
    fn test_predict_explained_single() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        let input = vec![1.0f32, 2.0];
        let (outputs, paths) = explainable.predict_explained(&input, 1);
        assert_eq!(outputs.len(), 1);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_predict_explained_batch() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let (outputs, paths) = explainable.predict_explained(&input, 2);
        assert_eq!(outputs.len(), 2);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_explain_one() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        let input = vec![1.0f32, 2.0];
        let _path = explainable.explain_one(&input);
    }

    #[test]
    fn test_contributions_match_prediction() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        let input = vec![2.0f32, 3.0];
        let (outputs, _) = explainable.predict_explained(&input, 1);
        let coeffs = explainable.compute_contributions(&input);
        let intercept = 0.0f32; // approximate
        let manual_sum: f32 = coeffs.iter().sum::<f32>() + intercept;
        // The prediction should be close to the sum of contributions + intercept
        assert!((outputs[0] - manual_sum).abs() < 5.0);
    }

    #[test]
    #[should_panic(expected = "Input length")]
    fn test_predict_explained_wrong_size() {
        let model = make_model();
        let explainable = LinearExplainable::new(model);
        let input = vec![1.0f32]; // too short for n_features=2, n_samples=1
        let _ = explainable.predict_explained(&input, 1);
    }
}
