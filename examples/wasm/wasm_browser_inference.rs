//! # Recipe: Browser Inference with WASM
//!
//! **Category**: WASM/Browser
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
//! 6. [x] WASM compatible (Verified)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Run model inference entirely in the browser via WASM.
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_browser_inference
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_browser_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Browser inference simulation (WASM-compatible)");
    println!();

    // Initialize WASM-compatible model
    let model = WasmModel::new(WasmModelConfig {
        name: "classifier".to_string(),
        input_size: 4,
        hidden_size: 8,
        output_size: 3,
    });

    ctx.record_metric("input_size", model.config.input_size as i64);
    ctx.record_metric("output_size", model.config.output_size as i64);

    println!("Model Configuration:");
    println!("  Name: {}", model.config.name);
    println!("  Input: {} features", model.config.input_size);
    println!("  Hidden: {} units", model.config.hidden_size);
    println!("  Output: {} classes", model.config.output_size);
    println!();

    // Simulate browser input
    let inputs = vec![0.5f32, 0.3, 0.8, 0.2];
    println!("Input features: {:?}", inputs);

    // Run inference
    let outputs = model.predict(&inputs)?;

    println!("Output probabilities: {:?}", outputs);

    // Find predicted class
    let predicted_class = outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    ctx.record_metric("predicted_class", predicted_class as i64);
    ctx.record_float_metric("confidence", f64::from(outputs[predicted_class]));

    println!();
    println!("Prediction:");
    println!("  Class: {}", predicted_class);
    println!("  Confidence: {:.2}%", outputs[predicted_class] * 100.0);

    // Performance metrics
    let perf = model.get_performance_metrics();
    println!();
    println!("Performance (simulated):");
    println!("  Inference time: {}ms", perf.inference_time_ms);
    println!("  Memory usage: {}KB", perf.memory_kb);
    println!("  WASM module size: {}KB", perf.wasm_size_kb);

    // Save inference result
    let result_path = ctx.path("inference_result.json");
    save_result(&result_path, &inputs, &outputs, predicted_class)?;
    println!();
    println!("Result saved to: {:?}", result_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmModelConfig {
    name: String,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

#[derive(Debug)]
struct WasmModel {
    config: WasmModelConfig,
    weights_hidden: Vec<Vec<f32>>,
    weights_output: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    inference_time_ms: u32,
    memory_kb: u32,
    wasm_size_kb: u32,
}

impl WasmModel {
    fn new(config: WasmModelConfig) -> Self {
        // Initialize deterministic weights
        let seed = hash_name_to_seed(&config.name);

        let weights_hidden = (0..config.hidden_size)
            .map(|i| {
                (0..config.input_size)
                    .map(|j| {
                        let idx = (seed as usize + i * config.input_size + j) % 100;
                        (idx as f32 - 50.0) / 100.0
                    })
                    .collect()
            })
            .collect();

        let weights_output = (0..config.output_size)
            .map(|i| {
                (0..config.hidden_size)
                    .map(|j| {
                        let idx = (seed as usize + i * config.hidden_size + j + 1000) % 100;
                        (idx as f32 - 50.0) / 100.0
                    })
                    .collect()
            })
            .collect();

        Self {
            config,
            weights_hidden,
            weights_output,
        }
    }

    fn predict(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        if inputs.len() != self.config.input_size {
            return Err(CookbookError::invalid_format(format!(
                "Expected {} inputs, got {}",
                self.config.input_size,
                inputs.len()
            )));
        }

        // Hidden layer (ReLU activation)
        let hidden: Vec<f32> = self
            .weights_hidden
            .iter()
            .map(|weights| {
                let sum: f32 = weights.iter().zip(inputs.iter()).map(|(w, x)| w * x).sum();
                sum.max(0.0) // ReLU
            })
            .collect();

        // Output layer (raw scores)
        let scores: Vec<f32> = self
            .weights_output
            .iter()
            .map(|weights| weights.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum())
            .collect();

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        Ok(exp_scores.iter().map(|e| e / sum_exp).collect())
    }

    fn get_performance_metrics(&self) -> PerformanceMetrics {
        let param_count = self.config.input_size * self.config.hidden_size
            + self.config.hidden_size * self.config.output_size;

        PerformanceMetrics {
            inference_time_ms: 1 + (param_count / 100) as u32,
            memory_kb: (param_count * 4 / 1024) as u32 + 10,
            wasm_size_kb: 50 + (param_count / 200) as u32,
        }
    }
}

fn save_result(
    path: &std::path::Path,
    inputs: &[f32],
    outputs: &[f32],
    predicted_class: usize,
) -> Result<()> {
    #[derive(Serialize)]
    struct Result<'a> {
        inputs: &'a [f32],
        outputs: &'a [f32],
        predicted_class: usize,
    }

    let result = Result {
        inputs,
        outputs,
        predicted_class,
    };

    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = WasmModel::new(WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        });

        assert_eq!(model.weights_hidden.len(), 8);
        assert_eq!(model.weights_output.len(), 3);
    }

    #[test]
    fn test_predict() {
        let model = WasmModel::new(WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        });

        let outputs = model.predict(&[0.5, 0.3, 0.8, 0.2]).unwrap();

        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let model = WasmModel::new(WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        });

        let outputs = model.predict(&[0.5, 0.3, 0.8, 0.2]).unwrap();
        let sum: f32 = outputs.iter().sum();

        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_deterministic_output() {
        let config = WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        };

        let model1 = WasmModel::new(config.clone());
        let model2 = WasmModel::new(config);

        let inputs = vec![0.5, 0.3, 0.8, 0.2];
        let out1 = model1.predict(&inputs).unwrap();
        let out2 = model2.predict(&inputs).unwrap();

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_wrong_input_size() {
        let model = WasmModel::new(WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        });

        let result = model.predict(&[0.5, 0.3]); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_metrics() {
        let model = WasmModel::new(WasmModelConfig {
            name: "test".to_string(),
            input_size: 4,
            hidden_size: 8,
            output_size: 3,
        });

        let perf = model.get_performance_metrics();

        assert!(perf.inference_time_ms > 0);
        assert!(perf.memory_kb > 0);
        assert!(perf.wasm_size_kb > 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_output_sums_to_one(inputs in proptest::collection::vec(-1.0f32..1.0, 4..5)) {
            let model = WasmModel::new(WasmModelConfig {
                name: "test".to_string(),
                input_size: 4,
                hidden_size: 8,
                output_size: 3,
            });

            if inputs.len() == 4 {
                let outputs = model.predict(&inputs).unwrap();
                let sum: f32 = outputs.iter().sum();
                prop_assert!((sum - 1.0).abs() < 0.01);
            }
        }

        #[test]
        fn prop_outputs_non_negative(inputs in proptest::collection::vec(-1.0f32..1.0, 4..5)) {
            let model = WasmModel::new(WasmModelConfig {
                name: "test".to_string(),
                input_size: 4,
                hidden_size: 8,
                output_size: 3,
            });

            if inputs.len() == 4 {
                let outputs = model.predict(&inputs).unwrap();
                for output in outputs {
                    prop_assert!(output >= 0.0);
                }
            }
        }
    }
}
