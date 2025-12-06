//! # Recipe: API Model Inference Call
//!
//! **Category**: API Integration
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
//! Call model inference via REST API (mock).
//!
//! ## Run Command
//! ```bash
//! cargo run --example api_call_model_inference
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("api_call_model_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Calling model inference via REST API (mock)");
    println!();

    // Configure API endpoint
    let config = ApiConfig {
        base_url: "http://localhost:8080".to_string(),
        model_name: "fraud-detector".to_string(),
        timeout_ms: 5000,
    };

    // Create inference request
    let request = InferenceRequest {
        inputs: vec![0.5, 0.3, 0.8, 0.1, 0.9],
        parameters: InferenceParameters {
            temperature: 1.0,
            max_tokens: 100,
        },
    };

    ctx.record_metric("input_size", request.inputs.len() as i64);

    // Display request
    println!("Request:");
    println!(
        "  Endpoint: {}/v1/models/{}/infer",
        config.base_url, config.model_name
    );
    println!("  Inputs: {:?}", request.inputs);
    println!();

    // Make mock API call
    let response = mock_api_call(&config, &request)?;

    ctx.record_metric("output_size", response.outputs.len() as i64);
    ctx.record_metric("latency_ms", i64::from(response.latency_ms));

    // Display response
    println!("Response:");
    println!("  Status: {}", response.status);
    println!("  Outputs: {:?}", response.outputs);
    println!("  Latency: {}ms", response.latency_ms);
    println!("  Model version: {}", response.model_version);

    // Save request/response for debugging
    let log_path = ctx.path("api_call.json");
    save_api_log(&log_path, &request, &response)?;
    println!();
    println!("API log saved to: {:?}", log_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiConfig {
    base_url: String,
    model_name: String,
    timeout_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceRequest {
    inputs: Vec<f32>,
    parameters: InferenceParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceParameters {
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceResponse {
    status: String,
    outputs: Vec<f32>,
    latency_ms: u32,
    model_version: String,
}

/// Mock API call (simulates network request)
fn mock_api_call(_config: &ApiConfig, request: &InferenceRequest) -> Result<InferenceResponse> {
    // Simulate processing
    let outputs: Vec<f32> = request.inputs.iter().map(|x| (x * 2.0).tanh()).collect();

    // Simulate latency (deterministic for testing)
    let latency_ms = 42 + request.inputs.len() as u32;

    Ok(InferenceResponse {
        status: "success".to_string(),
        outputs,
        latency_ms,
        model_version: "1.2.0".to_string(),
    })
}

fn save_api_log(
    path: &std::path::Path,
    request: &InferenceRequest,
    response: &InferenceResponse,
) -> Result<()> {
    #[derive(Serialize)]
    struct ApiLog<'a> {
        request: &'a InferenceRequest,
        response: &'a InferenceResponse,
    }

    let log = ApiLog { request, response };
    let json = serde_json::to_string_pretty(&log)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_api_call() {
        let config = ApiConfig {
            base_url: "http://localhost".to_string(),
            model_name: "test".to_string(),
            timeout_ms: 1000,
        };

        let request = InferenceRequest {
            inputs: vec![0.5, 0.5],
            parameters: InferenceParameters {
                temperature: 1.0,
                max_tokens: 10,
            },
        };

        let response = mock_api_call(&config, &request).unwrap();

        assert_eq!(response.status, "success");
        assert_eq!(response.outputs.len(), 2);
    }

    #[test]
    fn test_output_transformation() {
        let config = ApiConfig {
            base_url: "http://localhost".to_string(),
            model_name: "test".to_string(),
            timeout_ms: 1000,
        };

        let request = InferenceRequest {
            inputs: vec![0.0],
            parameters: InferenceParameters {
                temperature: 1.0,
                max_tokens: 10,
            },
        };

        let response = mock_api_call(&config, &request).unwrap();

        // tanh(0) = 0
        assert!((response.outputs[0] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_api_log_save() {
        let ctx = RecipeContext::new("test_api_log").unwrap();
        let path = ctx.path("log.json");

        let request = InferenceRequest {
            inputs: vec![1.0],
            parameters: InferenceParameters {
                temperature: 1.0,
                max_tokens: 10,
            },
        };

        let response = InferenceResponse {
            status: "success".to_string(),
            outputs: vec![0.96],
            latency_ms: 50,
            model_version: "1.0.0".to_string(),
        };

        save_api_log(&path, &request, &response).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_deterministic_latency() {
        let config = ApiConfig {
            base_url: "http://localhost".to_string(),
            model_name: "test".to_string(),
            timeout_ms: 1000,
        };

        let request = InferenceRequest {
            inputs: vec![1.0, 2.0, 3.0],
            parameters: InferenceParameters {
                temperature: 1.0,
                max_tokens: 10,
            },
        };

        let r1 = mock_api_call(&config, &request).unwrap();
        let r2 = mock_api_call(&config, &request).unwrap();

        assert_eq!(r1.latency_ms, r2.latency_ms);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_output_size_matches_input(inputs in proptest::collection::vec(-1.0f32..1.0, 1..100)) {
            let config = ApiConfig {
                base_url: "http://localhost".to_string(),
                model_name: "test".to_string(),
                timeout_ms: 1000,
            };

            let request = InferenceRequest {
                inputs: inputs.clone(),
                parameters: InferenceParameters {
                    temperature: 1.0,
                    max_tokens: 10,
                },
            };

            let response = mock_api_call(&config, &request).unwrap();
            prop_assert_eq!(response.outputs.len(), inputs.len());
        }

        #[test]
        fn prop_outputs_bounded(inputs in proptest::collection::vec(-10.0f32..10.0, 1..50)) {
            let config = ApiConfig {
                base_url: "http://localhost".to_string(),
                model_name: "test".to_string(),
                timeout_ms: 1000,
            };

            let request = InferenceRequest {
                inputs,
                parameters: InferenceParameters {
                    temperature: 1.0,
                    max_tokens: 10,
                },
            };

            let response = mock_api_call(&config, &request).unwrap();

            // tanh output is bounded in (-1, 1)
            for &output in &response.outputs {
                prop_assert!(output >= -1.0 && output <= 1.0);
            }
        }
    }
}
