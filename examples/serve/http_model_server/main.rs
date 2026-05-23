#![allow(unused_imports)]
//! HTTP Model Server Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates model serving patterns: request parsing, batched inference,
//! health checks, request routing, and metrics collection. Simulates an
//! HTTP server without external dependencies.
//!
//! # API Endpoints
//!
//! ```text
//! POST /v1/predict      - Single inference request
//! POST /v1/batch        - Batched inference
//! GET  /health          - Health check
//! GET  /metrics         - Prometheus-style metrics
//! GET  /v1/models       - List loaded models
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example http_model_server
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr serve model.apr          # APR native format
//! apr serve model.gguf         # GGUF (llama.cpp compatible)
//! apr serve model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== HTTP Model Server Example ===\n");

    let mut server = ModelServer::new();

    // =========================================================================
    // Section 1: Model Loading
    // =========================================================================
    println!("1. Loading Models");
    println!("   ─────────────────────────────────────────");

    server.load_model("default", "1.0.0", 42);
    server.load_model("classifier", "2.1.0", 100);
    server.load_model("embedder", "1.5.0", 200);

    println!("   Loaded 3 models:");
    for (name, model) in &server.models {
        println!(
            "     {} v{} ({}x{} weights)",
            name, model.version, OUTPUT_DIM, INPUT_DIM
        );
    }
    println!();

    // =========================================================================
    // Section 2: Health Check
    // =========================================================================
    println!("2. Health Check");
    println!("   ─────────────────────────────────────────");

    let resp = server.handle_request(&Request {
        method: "GET",
        path: "/health".to_string(),
        body: None,
    });
    println!("   GET /health → {} {}", resp.status, resp.body);
    println!();

    // =========================================================================
    // Section 3: Single Predictions
    // =========================================================================
    println!("3. Single Predictions");
    println!("   ─────────────────────────────────────────");

    let requests = [
        ("default model", "{\"model\":\"default\",\"input\":[...]}"),
        ("classifier", "{\"model\":\"classifier\",\"input\":[...]}"),
        ("unknown model", "{\"model\":\"unknown\",\"input\":[...]}"),
    ];

    for &(desc, body) in &requests {
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/predict".to_string(),
            body: Some((*body).to_string()),
        });
        println!(
            "   {} → {} ({}us) {}",
            desc,
            resp.status,
            resp.latency_us,
            &resp.body[..resp.body.len().min(60)]
        );
    }
    println!();

    // =========================================================================
    // Section 4: Batch Inference
    // =========================================================================
    println!("4. Batch Inference");
    println!("   ─────────────────────────────────────────");

    for batch_size in [1, 4, 8, 16] {
        let inputs_str = "input,".repeat(batch_size);
        let body = format!("{{\"model\":\"classifier\",\"inputs\":[{}]}}", inputs_str);
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/batch".to_string(),
            body: Some(body),
        });
        println!(
            "   batch_size={:>2} → {} ({}us)",
            batch_size, resp.status, resp.latency_us
        );
    }
    println!();

    // =========================================================================
    // Section 5: Error Handling
    // =========================================================================
    println!("5. Error Handling");
    println!("   ─────────────────────────────────────────");

    let error_requests = [
        ("GET", "/unknown", None, "Unknown endpoint"),
        ("POST", "/v1/predict", None, "Missing body"),
        (
            "POST",
            "/v1/predict",
            Some("{\"model\":\"nonexistent\"}"),
            "Unknown model",
        ),
    ];

    for &(method, path, body, desc) in &error_requests {
        let resp = server.handle_request(&Request {
            method,
            path: (*path).to_string(),
            body: body.map(ToString::to_string),
        });
        println!("   {} → {} {}", desc, resp.status, resp.body);
    }
    println!();

    // =========================================================================
    // Section 6: Throughput Benchmark
    // =========================================================================
    println!("6. Throughput Benchmark");
    println!("   ─────────────────────────────────────────");

    let n_requests = 1000;
    let start = Instant::now();
    for i in 0..n_requests {
        let body = format!("{{\"model\":\"default\",\"input_id\":{}}}", i);
        server.handle_request(&Request {
            method: "POST",
            path: "/v1/predict".to_string(),
            body: Some(body),
        });
    }
    let elapsed = start.elapsed();
    let rps = f64::from(n_requests) / elapsed.as_secs_f64();

    println!("   Requests:    {n_requests}");
    println!("   Total time:  {} ms", elapsed.as_millis());
    println!("   Throughput:  {rps:.0} req/sec");
    println!(
        "   Avg latency: {:.1} us",
        elapsed.as_micros() as f64 / f64::from(n_requests)
    );
    println!();

    // =========================================================================
    // Section 7: Metrics
    // =========================================================================
    println!("7. Server Metrics");
    println!("   ─────────────────────────────────────────");

    let resp = server.handle_request(&Request {
        method: "GET",
        path: "/metrics".to_string(),
        body: None,
    });
    for line in resp.body.lines() {
        if !line.starts_with('#') {
            println!("   {}", line);
        }
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_predict_dimensions() {
        let model = ServedModel::new("test", "1.0", 42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.predict(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_model_predict_deterministic() {
        let model = ServedModel::new("test", "1.0", 42);
        let input = vec![0.1; INPUT_DIM];
        let o1 = model.predict(&input);
        let o2 = model.predict(&input);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_server_health() {
        let mut server = ModelServer::new();
        server.load_model("test", "1.0", 42);
        let resp = server.handle_request(&Request {
            method: "GET",
            path: "/health".to_string(),
            body: None,
        });
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("healthy"));
    }

    #[test]
    fn test_server_predict() {
        let mut server = ModelServer::new();
        server.load_model("default", "1.0", 42);
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/predict".to_string(),
            body: Some("{\"model\":\"default\"}".to_string()),
        });
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn test_server_404() {
        let mut server = ModelServer::new();
        let resp = server.handle_request(&Request {
            method: "GET",
            path: "/nonexistent".to_string(),
            body: None,
        });
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_server_missing_body() {
        let mut server = ModelServer::new();
        server.load_model("default", "1.0", 42);
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/predict".to_string(),
            body: None,
        });
        assert_eq!(resp.status, 400);
    }

    #[test]
    fn test_server_unknown_model() {
        let mut server = ModelServer::new();
        // Only load "classifier" so "default" model is missing
        server.load_model("classifier", "1.0", 42);
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/predict".to_string(),
            body: Some("{\"model\":\"unknown\"}".to_string()),
        });
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut server = ModelServer::new();
        server.load_model("default", "1.0", 42);

        for _ in 0..5 {
            server.handle_request(&Request {
                method: "POST",
                path: "/v1/predict".to_string(),
                body: Some("{\"model\":\"default\"}".to_string()),
            });
        }

        assert_eq!(server.metrics.predictions_total, 5);
        assert!(server.metrics.requests_total >= 5);
    }

    #[test]
    fn test_batch_predict() {
        let mut server = ModelServer::new();
        server.load_model("classifier", "1.0", 42);
        let resp = server.handle_request(&Request {
            method: "POST",
            path: "/v1/batch".to_string(),
            body: Some("{\"model\":\"classifier\",\"inputs\":[input,input,input]}".to_string()),
        });
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("batch_size"));
    }
}
