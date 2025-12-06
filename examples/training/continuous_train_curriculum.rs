//! # Recipe: Curriculum Learning
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
//! Implement curriculum learning with progressive difficulty.
//!
//! ## Run Command
//! ```bash
//! cargo run --example continuous_train_curriculum
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("continuous_train_curriculum")?;

    let n_features = 3;
    let n_stages = 4;
    let samples_per_stage = 200;
    let learning_rate = 0.02f32;

    ctx.record_metric("n_features", n_features as i64);
    ctx.record_metric("n_stages", n_stages as i64);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Curriculum Learning with Progressive Difficulty");
    println!();

    // Initialize model
    let mut weights = vec![0.0f32; n_features];
    let mut bias = 0.0f32;

    // Curriculum: start easy, increase difficulty
    for stage in 0..n_stages {
        let difficulty = stage + 1;
        let noise_level = 0.05 * difficulty as f32;

        let stage_seed = hash_name_to_seed(&format!("stage_{}", stage));
        let (x, y) =
            generate_curriculum_data(stage_seed, samples_per_stage, n_features, difficulty);

        // Train on this stage
        let stage_loss = train_stage(&x, &y, &mut weights, &mut bias, n_features, learning_rate);

        println!(
            "Stage {} (difficulty={}): loss={:.4}, noise={:.2}",
            stage + 1,
            difficulty,
            stage_loss,
            noise_level
        );

        ctx.record_float_metric(&format!("stage_{}_loss", stage + 1), stage_loss);

        // Save stage checkpoint
        let checkpoint_path = ctx.path(&format!("curriculum_stage_{}.apr", stage + 1));
        save_checkpoint(&checkpoint_path, &weights, bias)?;
    }

    // Final evaluation on hard data
    let eval_seed = hash_name_to_seed("curriculum_eval");
    let (x_eval, y_eval) = generate_curriculum_data(eval_seed, 100, n_features, n_stages);
    let final_loss = evaluate(&x_eval, &y_eval, &weights, bias, n_features);

    ctx.record_float_metric("final_loss", final_loss);

    let model_path = ctx.path("curriculum_final.apr");
    save_checkpoint(&model_path, &weights, bias)?;

    println!();
    println!("Curriculum training complete:");
    println!("  Final weights: {:?}", weights);
    println!("  Final bias: {:.4}", bias);
    println!("  Final loss (hard data): {:.4}", final_loss);
    println!("  Model saved to: {:?}", model_path);

    Ok(())
}

/// Generate curriculum data with specified difficulty
fn generate_curriculum_data(
    seed: u64,
    n_samples: usize,
    n_features: usize,
    difficulty: usize,
) -> (Vec<f32>, Vec<f32>) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Difficulty affects:
    // 1. Noise level
    // 2. Data range (harder = wider range)
    // 3. Number of active features
    let noise_level = 0.05 * difficulty as f32;
    let data_range = 1.0 + 0.5 * difficulty as f32;
    let active_features = (n_features.min(difficulty)).max(1);

    // True weights (only active_features have non-zero weights)
    let true_weights: Vec<f32> = (0..n_features)
        .map(|i| {
            if i < active_features {
                (i + 1) as f32
            } else {
                0.0
            }
        })
        .collect();

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x: Vec<f32> = (0..n_features)
            .map(|_| rng.gen_range(-data_range..data_range))
            .collect();

        let mut y = 0.5f32;
        for (&xi, &wi) in x.iter().zip(true_weights.iter()) {
            y += xi * wi;
        }
        y += rng.gen_range(-noise_level..noise_level);

        x_data.extend(x);
        y_data.push(y);
    }

    (x_data, y_data)
}

/// Train on a curriculum stage
fn train_stage(
    x_data: &[f32],
    y_data: &[f32],
    weights: &mut [f32],
    bias: &mut f32,
    n_features: usize,
    learning_rate: f32,
) -> f64 {
    let n_samples = y_data.len();
    let epochs = 10;
    let mut final_loss = 0.0f64;

    for _ in 0..epochs {
        final_loss = 0.0;
        for i in 0..n_samples {
            let mut pred = *bias;
            for j in 0..n_features {
                pred += weights[j] * x_data[i * n_features + j];
            }

            let error = pred - y_data[i];
            final_loss += f64::from(error).powi(2);

            for j in 0..n_features {
                weights[j] -= learning_rate * error * x_data[i * n_features + j] / n_samples as f32;
            }
            *bias -= learning_rate * error / n_samples as f32;
        }
        final_loss /= n_samples as f64;
    }

    final_loss
}

fn evaluate(x_data: &[f32], y_data: &[f32], weights: &[f32], bias: f32, n_features: usize) -> f64 {
    let n_samples = y_data.len();
    let mut total_loss = 0.0f64;

    for i in 0..n_samples {
        let mut pred = bias;
        for j in 0..n_features {
            pred += weights[j] * x_data[i * n_features + j];
        }
        total_loss += f64::from(pred - y_data[i]).powi(2);
    }

    total_loss / n_samples as f64
}

fn save_checkpoint(path: &std::path::Path, weights: &[f32], bias: f32) -> Result<()> {
    let mut converter = AprConverter::new();
    converter.add_tensor(TensorData {
        name: "weights".to_string(),
        shape: vec![weights.len()],
        dtype: DataType::F32,
        data: weights.iter().flat_map(|f| f.to_le_bytes()).collect(),
    });
    converter.add_tensor(TensorData {
        name: "bias".to_string(),
        shape: vec![1],
        dtype: DataType::F32,
        data: bias.to_le_bytes().to_vec(),
    });

    std::fs::write(path, converter.to_apr()?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_data_generation() {
        let (x, y) = generate_curriculum_data(42, 50, 4, 2);
        assert_eq!(x.len(), 200);
        assert_eq!(y.len(), 50);
    }

    #[test]
    fn test_difficulty_affects_data_range() {
        // Higher difficulty = wider data range
        let (x_easy, _) = generate_curriculum_data(42, 100, 2, 1);
        let (x_hard, _) = generate_curriculum_data(42, 100, 2, 4);

        let max_easy = x_easy.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_hard = x_hard.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // Hard data should have wider range
        assert!(max_hard >= max_easy);
    }

    #[test]
    fn test_stage_training() {
        let (x, y) = generate_curriculum_data(42, 100, 3, 1);
        let mut weights = vec![0.0f32; 3];
        let mut bias = 0.0f32;

        let loss = train_stage(&x, &y, &mut weights, &mut bias, 3, 0.1);

        assert!(loss >= 0.0);
        assert!(weights.iter().any(|&w| w.abs() > 0.01));
    }

    #[test]
    fn test_deterministic() {
        let (x1, y1) = generate_curriculum_data(42, 50, 3, 2);
        let (x2, y2) = generate_curriculum_data(42, 50, 3, 2);

        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_loss_non_negative(difficulty in 1usize..5) {
            let (x, y) = generate_curriculum_data(42, 50, 3, difficulty);
            let mut weights = vec![0.0f32; 3];
            let mut bias = 0.0f32;

            let loss = train_stage(&x, &y, &mut weights, &mut bias, 3, 0.1);
            prop_assert!(loss >= 0.0);
        }

        #[test]
        fn prop_data_sizes(n_samples in 10usize..100, n_features in 1usize..10) {
            let (x, y) = generate_curriculum_data(42, n_samples, n_features, 2);
            prop_assert_eq!(x.len(), n_samples * n_features);
            prop_assert_eq!(y.len(), n_samples);
        }
    }
}
