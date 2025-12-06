//! # Recipe: Lambda Inference Function
//!
//! **Category**: Serverless/Lambda
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
//! Deploy model inference as AWS Lambda function (simulated).
//!
//! ## Run Command
//! ```bash
//! cargo run --example serverless_lambda_inference
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("serverless_lambda_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Lambda inference function simulation");
    println!();

    // Create Lambda runtime context
    let lambda_ctx = LambdaContext {
        function_name: "fraud-detector-lambda".to_string(),
        function_version: "$LATEST".to_string(),
        memory_limit_mb: 512,
        timeout_seconds: 30,
        request_id: "req-abc123".to_string(),
    };

    println!("Lambda Context:");
    println!("  Function: {}", lambda_ctx.function_name);
    println!("  Version: {}", lambda_ctx.function_version);
    println!("  Memory: {}MB", lambda_ctx.memory_limit_mb);
    println!("  Timeout: {}s", lambda_ctx.timeout_seconds);
    println!();

    // Simulate Lambda invocation
    let event = LambdaEvent {
        body: InferenceRequest {
            inputs: vec![0.5, 0.3, 0.8, 0.1],
        },
        request_context: RequestContext {
            stage: "prod".to_string(),
            path: "/infer".to_string(),
        },
    };

    ctx.record_metric("input_size", event.body.inputs.len() as i64);

    println!("Event:");
    println!("  Inputs: {:?}", event.body.inputs);
    println!("  Stage: {}", event.request_context.stage);
    println!();

    // Handler execution
    let response = handler(&lambda_ctx, &event)?;

    ctx.record_metric("status_code", i64::from(response.status_code));
    ctx.record_float_metric("billed_duration_ms", f64::from(response.billed_duration_ms));

    println!("Response:");
    println!("  Status: {}", response.status_code);
    println!("  Body: {}", response.body);
    println!("  Billed duration: {}ms", response.billed_duration_ms);

    // Save Lambda metrics
    let metrics_path = ctx.path("lambda_metrics.json");
    save_metrics(&metrics_path, &lambda_ctx, &response)?;
    println!();
    println!("Metrics saved to: {:?}", metrics_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LambdaContext {
    function_name: String,
    function_version: String,
    memory_limit_mb: u32,
    timeout_seconds: u32,
    request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LambdaEvent {
    body: InferenceRequest,
    request_context: RequestContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceRequest {
    inputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestContext {
    stage: String,
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LambdaResponse {
    status_code: u16,
    body: String,
    billed_duration_ms: u32,
    memory_used_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceOutput {
    predictions: Vec<f32>,
    model_version: String,
}

fn handler(ctx: &LambdaContext, event: &LambdaEvent) -> Result<LambdaResponse> {
    // Simulate model inference
    let predictions: Vec<f32> = event.body.inputs.iter().map(|x| (x * 2.0).tanh()).collect();

    let output = InferenceOutput {
        predictions,
        model_version: "1.0.0".to_string(),
    };

    let body =
        serde_json::to_string(&output).map_err(|e| CookbookError::Serialization(e.to_string()))?;

    // Deterministic billing calculation
    let billed_duration = 10 + event.body.inputs.len() as u32 * 5;
    let memory_used = ctx.memory_limit_mb / 2;

    Ok(LambdaResponse {
        status_code: 200,
        body,
        billed_duration_ms: billed_duration,
        memory_used_mb: memory_used,
    })
}

fn save_metrics(
    path: &std::path::Path,
    ctx: &LambdaContext,
    response: &LambdaResponse,
) -> Result<()> {
    #[derive(Serialize)]
    struct Metrics<'a> {
        function: &'a str,
        request_id: &'a str,
        status_code: u16,
        billed_duration_ms: u32,
        memory_used_mb: u32,
        memory_limit_mb: u32,
    }

    let metrics = Metrics {
        function: &ctx.function_name,
        request_id: &ctx.request_id,
        status_code: response.status_code,
        billed_duration_ms: response.billed_duration_ms,
        memory_used_mb: response.memory_used_mb,
        memory_limit_mb: ctx.memory_limit_mb,
    };

    let json = serde_json::to_string_pretty(&metrics)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_success() {
        let ctx = LambdaContext {
            function_name: "test".to_string(),
            function_version: "1".to_string(),
            memory_limit_mb: 256,
            timeout_seconds: 10,
            request_id: "req-1".to_string(),
        };

        let event = LambdaEvent {
            body: InferenceRequest {
                inputs: vec![0.5, 0.5],
            },
            request_context: RequestContext {
                stage: "test".to_string(),
                path: "/".to_string(),
            },
        };

        let response = handler(&ctx, &event).unwrap();

        assert_eq!(response.status_code, 200);
        assert!(response.body.contains("predictions"));
    }

    #[test]
    fn test_deterministic_billing() {
        let ctx = LambdaContext {
            function_name: "test".to_string(),
            function_version: "1".to_string(),
            memory_limit_mb: 256,
            timeout_seconds: 10,
            request_id: "req-1".to_string(),
        };

        let event = LambdaEvent {
            body: InferenceRequest {
                inputs: vec![1.0, 2.0, 3.0],
            },
            request_context: RequestContext {
                stage: "test".to_string(),
                path: "/".to_string(),
            },
        };

        let r1 = handler(&ctx, &event).unwrap();
        let r2 = handler(&ctx, &event).unwrap();

        assert_eq!(r1.billed_duration_ms, r2.billed_duration_ms);
    }

    #[test]
    fn test_memory_usage() {
        let ctx = LambdaContext {
            function_name: "test".to_string(),
            function_version: "1".to_string(),
            memory_limit_mb: 512,
            timeout_seconds: 10,
            request_id: "req-1".to_string(),
        };

        let event = LambdaEvent {
            body: InferenceRequest { inputs: vec![1.0] },
            request_context: RequestContext {
                stage: "test".to_string(),
                path: "/".to_string(),
            },
        };

        let response = handler(&ctx, &event).unwrap();

        assert!(response.memory_used_mb <= ctx.memory_limit_mb);
    }

    #[test]
    fn test_save_metrics() {
        let recipe_ctx = RecipeContext::new("test_lambda_metrics").unwrap();
        let path = recipe_ctx.path("metrics.json");

        let ctx = LambdaContext {
            function_name: "test".to_string(),
            function_version: "1".to_string(),
            memory_limit_mb: 256,
            timeout_seconds: 10,
            request_id: "req-1".to_string(),
        };

        let response = LambdaResponse {
            status_code: 200,
            body: "{}".to_string(),
            billed_duration_ms: 10,
            memory_used_mb: 128,
        };

        save_metrics(&path, &ctx, &response).unwrap();
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
        fn prop_always_returns_200(inputs in proptest::collection::vec(-1.0f32..1.0, 1..20)) {
            let ctx = LambdaContext {
                function_name: "test".to_string(),
                function_version: "1".to_string(),
                memory_limit_mb: 256,
                timeout_seconds: 10,
                request_id: "req-1".to_string(),
            };

            let event = LambdaEvent {
                body: InferenceRequest { inputs },
                request_context: RequestContext {
                    stage: "test".to_string(),
                    path: "/".to_string(),
                },
            };

            let response = handler(&ctx, &event).unwrap();
            prop_assert_eq!(response.status_code, 200);
        }

        #[test]
        fn prop_billing_increases_with_inputs(n in 1usize..50) {
            let ctx = LambdaContext {
                function_name: "test".to_string(),
                function_version: "1".to_string(),
                memory_limit_mb: 256,
                timeout_seconds: 10,
                request_id: "req-1".to_string(),
            };

            let event = LambdaEvent {
                body: InferenceRequest { inputs: vec![1.0; n] },
                request_context: RequestContext {
                    stage: "test".to_string(),
                    path: "/".to_string(),
                },
            };

            let response = handler(&ctx, &event).unwrap();
            let expected = 10 + n as u32 * 5;
            prop_assert_eq!(response.billed_duration_ms, expected);
        }
    }
}
