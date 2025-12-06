//! # Recipe: Batch Model Inference
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
//! Process multiple inference requests in a batch for throughput.
//!
//! ## Run Command
//! ```bash
//! cargo run --example api_batch_inference
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("api_batch_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Batch inference processing");
    println!();

    // Create batch of requests
    let requests: Vec<BatchRequest> = (0..5)
        .map(|i| BatchRequest {
            id: format!("req-{:03}", i),
            inputs: vec![0.1 * i as f32, 0.2 * i as f32, 0.3 * i as f32],
        })
        .collect();

    ctx.record_metric("batch_size", requests.len() as i64);

    println!("Batch requests:");
    for req in &requests {
        println!("  {}: {:?}", req.id, req.inputs);
    }
    println!();

    // Process batch
    let batch_result = process_batch(&requests)?;

    ctx.record_metric("successful", batch_result.successful as i64);
    ctx.record_metric("failed", batch_result.failed as i64);
    ctx.record_metric("total_latency_ms", i64::from(batch_result.total_latency_ms));

    println!("Batch results:");
    for result in &batch_result.results {
        match &result.status {
            ResultStatus::Success { outputs } => {
                println!("  {} [OK]: {:?}", result.id, outputs);
            }
            ResultStatus::Error { message } => {
                println!("  {} [ERR]: {}", result.id, message);
            }
        }
    }

    println!();
    println!("Summary:");
    println!(
        "  Successful: {}/{}",
        batch_result.successful,
        requests.len()
    );
    println!("  Failed: {}", batch_result.failed);
    println!("  Total latency: {}ms", batch_result.total_latency_ms);
    println!(
        "  Avg latency/request: {:.1}ms",
        f64::from(batch_result.total_latency_ms) / requests.len() as f64
    );

    // Save batch results
    let results_path = ctx.path("batch_results.json");
    save_results(&results_path, &batch_result)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchRequest {
    id: String,
    inputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchResponse {
    id: String,
    status: ResultStatus,
    latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ResultStatus {
    Success { outputs: Vec<f32> },
    Error { message: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchResult {
    results: Vec<BatchResponse>,
    successful: usize,
    failed: usize,
    total_latency_ms: u32,
}

fn process_batch(requests: &[BatchRequest]) -> Result<BatchResult> {
    let mut results = Vec::with_capacity(requests.len());
    let mut successful = 0;
    let mut failed = 0;
    let mut total_latency = 0u32;

    for request in requests {
        let (response, latency) = process_single(request);
        total_latency += latency;

        match &response.status {
            ResultStatus::Success { .. } => successful += 1,
            ResultStatus::Error { .. } => failed += 1,
        }

        results.push(response);
    }

    Ok(BatchResult {
        results,
        successful,
        failed,
        total_latency_ms: total_latency,
    })
}

fn process_single(request: &BatchRequest) -> (BatchResponse, u32) {
    // Deterministic mock inference
    let outputs: Vec<f32> = request.inputs.iter().map(|x| (x * 2.0).tanh()).collect();

    // Deterministic latency based on input size
    let latency = 10 + request.inputs.len() as u32 * 2;

    let response = BatchResponse {
        id: request.id.clone(),
        status: ResultStatus::Success { outputs },
        latency_ms: latency,
    };

    (response, latency)
}

fn save_results(path: &std::path::Path, result: &BatchResult) -> Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processing() {
        let requests = vec![
            BatchRequest {
                id: "r1".to_string(),
                inputs: vec![1.0, 2.0],
            },
            BatchRequest {
                id: "r2".to_string(),
                inputs: vec![3.0, 4.0],
            },
        ];

        let result = process_batch(&requests).unwrap();

        assert_eq!(result.results.len(), 2);
        assert_eq!(result.successful, 2);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_single_processing() {
        let request = BatchRequest {
            id: "test".to_string(),
            inputs: vec![0.5],
        };

        let (response, latency) = process_single(&request);

        assert_eq!(response.id, "test");
        assert!(latency > 0);
        assert!(matches!(response.status, ResultStatus::Success { .. }));
    }

    #[test]
    fn test_output_transformation() {
        let request = BatchRequest {
            id: "test".to_string(),
            inputs: vec![0.0],
        };

        let (response, _) = process_single(&request);

        if let ResultStatus::Success { outputs } = response.status {
            assert!((outputs[0] - 0.0).abs() < 0.001); // tanh(0) = 0
        } else {
            panic!("Expected success");
        }
    }

    #[test]
    fn test_deterministic_latency() {
        let request = BatchRequest {
            id: "test".to_string(),
            inputs: vec![1.0, 2.0, 3.0],
        };

        let (_, latency1) = process_single(&request);
        let (_, latency2) = process_single(&request);

        assert_eq!(latency1, latency2);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_batch_save").unwrap();
        let path = ctx.path("results.json");

        let result = BatchResult {
            results: vec![],
            successful: 0,
            failed: 0,
            total_latency_ms: 0,
        };

        save_results(&path, &result).unwrap();
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
        fn prop_batch_size_matches(n in 1usize..20) {
            let requests: Vec<_> = (0..n)
                .map(|i| BatchRequest {
                    id: format!("r{}", i),
                    inputs: vec![i as f32],
                })
                .collect();

            let result = process_batch(&requests).unwrap();
            prop_assert_eq!(result.results.len(), n);
        }

        #[test]
        fn prop_all_successful(n in 1usize..10) {
            let requests: Vec<_> = (0..n)
                .map(|i| BatchRequest {
                    id: format!("r{}", i),
                    inputs: vec![i as f32],
                })
                .collect();

            let result = process_batch(&requests).unwrap();
            prop_assert_eq!(result.successful, n);
            prop_assert_eq!(result.failed, 0);
        }

        #[test]
        fn prop_outputs_bounded(inputs in proptest::collection::vec(-10.0f32..10.0, 1..10)) {
            let request = BatchRequest {
                id: "test".to_string(),
                inputs,
            };

            let (response, _) = process_single(&request);

            if let ResultStatus::Success { outputs } = response.status {
                for output in outputs {
                    prop_assert!(output >= -1.0 && output <= 1.0);
                }
            }
        }
    }
}
