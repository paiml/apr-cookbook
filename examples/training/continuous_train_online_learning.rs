//! # Recipe: Online Learning with Single-Sample Updates
//!
//! **Category**: Continuous Training
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
//! Implement online learning with single-sample gradient updates.
//!
//! ## Run Command
//! ```bash
//! cargo run --example continuous_train_online_learning
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("continuous_train_online_learning")?;

    let n_features = 3;
    let n_samples = 500;
    let learning_rate = 0.05f32;

    // Initialize model
    let mut model = OnlineModel::new(n_features);

    ctx.record_metric("n_features", n_features as i64);
    ctx.record_metric("n_samples", n_samples as i64);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Online learning with single-sample updates...");
    println!();

    // Stream samples one at a time
    let mut losses = Vec::with_capacity(n_samples);
    let stream_seed = hash_name_to_seed("online_stream");

    for i in 0..n_samples {
        // Generate single sample
        let sample_seed = stream_seed.wrapping_add(i as u64);
        let (x, y) = generate_single_sample(sample_seed, n_features);

        // Online update
        let loss = model.update(&x, y, learning_rate);
        losses.push(loss);

        // Log progress every 100 samples
        if (i + 1) % 100 == 0 {
            let avg_loss: f64 = losses.iter().skip(i.saturating_sub(99)).sum::<f64>() / 100.0;
            println!(
                "Sample {}: avg_loss={:.4}, weights={:?}",
                i + 1,
                avg_loss,
                model.weights
            );
        }
    }

    // Final metrics
    let final_loss: f64 = losses.iter().rev().take(50).sum::<f64>() / 50.0;
    ctx.record_float_metric("final_avg_loss", final_loss);

    // Save model
    let model_path = ctx.path("online_model.apr");
    model.save(&model_path)?;

    println!();
    println!("Training complete:");
    println!("  Total samples processed: {}", n_samples);
    println!("  Final weights: {:?}", model.weights);
    println!("  Final bias: {:.4}", model.bias);
    println!("  Final avg loss (last 50): {:.4}", final_loss);
    println!("  Model saved to: {:?}", model_path);

    Ok(())
}

/// Online learning model with single-sample updates
#[derive(Debug)]
struct OnlineModel {
    weights: Vec<f32>,
    bias: f32,
    n_updates: usize,
}

impl OnlineModel {
    fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0f32; n_features],
            bias: 0.0f32,
            n_updates: 0,
        }
    }

    /// Perform single-sample SGD update
    fn update(&mut self, x: &[f32], y: f32, learning_rate: f32) -> f64 {
        // Forward pass
        let pred = self.predict(x);
        let error = pred - y;
        let loss = f64::from(error).powi(2);

        // Backward pass
        for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
            *w -= learning_rate * error * xi;
        }
        self.bias -= learning_rate * error;

        self.n_updates += 1;
        loss
    }

    fn predict(&self, x: &[f32]) -> f32 {
        let mut pred = self.bias;
        for (&w, &xi) in self.weights.iter().zip(x.iter()) {
            pred += w * xi;
        }
        pred
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut converter = AprConverter::new();
        converter.set_metadata(ConversionMetadata {
            name: Some("online-model".to_string()),
            architecture: Some("linear-online".to_string()),
            source_format: None,
            custom: std::collections::HashMap::new(),
        });

        converter.add_tensor(TensorData {
            name: "weights".to_string(),
            shape: vec![self.weights.len()],
            dtype: DataType::F32,
            data: self.weights.iter().flat_map(|f| f.to_le_bytes()).collect(),
        });

        converter.add_tensor(TensorData {
            name: "bias".to_string(),
            shape: vec![1],
            dtype: DataType::F32,
            data: self.bias.to_le_bytes().to_vec(),
        });

        let apr_bytes = converter.to_apr()?;
        std::fs::write(path, apr_bytes)?;

        Ok(())
    }
}

/// Generate a single training sample
fn generate_single_sample(seed: u64, n_features: usize) -> (Vec<f32>, f32) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // True weights
    let true_weights: Vec<f32> = (0..n_features).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let true_bias = 1.0f32;

    let x: Vec<f32> = (0..n_features)
        .map(|_| rng.gen_range(-2.0f32..2.0f32))
        .collect();

    let mut y = true_bias;
    for (&xi, &wi) in x.iter().zip(true_weights.iter()) {
        y += xi * wi;
    }
    y += rng.gen_range(-0.1f32..0.1f32); // Noise

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = OnlineModel::new(5);
        assert_eq!(model.weights.len(), 5);
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.n_updates, 0);
    }

    #[test]
    fn test_single_update() {
        let mut model = OnlineModel::new(2);
        let x = vec![1.0f32, 2.0];
        let y = 3.0f32;

        let loss = model.update(&x, y, 0.1);
        assert!(loss >= 0.0);
        assert_eq!(model.n_updates, 1);
    }

    #[test]
    fn test_prediction() {
        let mut model = OnlineModel::new(2);
        model.weights = vec![1.0, 2.0];
        model.bias = 0.5;

        let x = vec![1.0f32, 1.0];
        let pred = model.predict(&x);

        // 0.5 + 1*1 + 2*1 = 3.5
        assert!((pred - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_learning() {
        let mut model = OnlineModel::new(2);

        // Train on consistent data
        let mut total_loss = 0.0f64;
        for i in 0..100 {
            let (x, y) = generate_single_sample(i as u64, 2);
            total_loss += model.update(&x, y, 0.1);
        }

        let avg_loss = total_loss / 100.0;

        // Should have learned something
        assert!(model.weights.iter().any(|&w| w.abs() > 0.01));
        assert!(avg_loss < 100.0);
    }

    #[test]
    fn test_deterministic_samples() {
        let (x1, y1) = generate_single_sample(42, 3);
        let (x2, y2) = generate_single_sample(42, 3);

        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_model_save() {
        let ctx = RecipeContext::new("test_online_save").unwrap();
        let mut model = OnlineModel::new(3);
        model.weights = vec![1.0, 2.0, 3.0];
        model.bias = 0.5;

        let path = ctx.path("model.apr");
        model.save(&path).unwrap();

        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_loss_non_negative(seed in 0u64..1000) {
            let mut model = OnlineModel::new(3);
            let (x, y) = generate_single_sample(seed, 3);
            let loss = model.update(&x, y, 0.1);
            prop_assert!(loss >= 0.0);
        }

        #[test]
        fn prop_update_count(n_updates in 1usize..100) {
            let mut model = OnlineModel::new(2);
            for i in 0..n_updates {
                let (x, y) = generate_single_sample(i as u64, 2);
                model.update(&x, y, 0.1);
            }
            prop_assert_eq!(model.n_updates, n_updates);
        }
    }
}
