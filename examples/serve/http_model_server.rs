//! HTTP Model Server Example
//!
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
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const INPUT_DIM: usize = 32;
const OUTPUT_DIM: usize = 8;

/// Model weights for serving
struct ServedModel {
    name: String,
    version: String,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl ServedModel {
    fn new(name: &str, version: &str, seed: u64) -> Self {
        let weights: Vec<f32> = (0..OUTPUT_DIM * INPUT_DIM)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (seed, "w", i).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
            })
            .collect();
        let bias: Vec<f32> = (0..OUTPUT_DIM)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (seed, "b", i).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.01
            })
            .collect();
        Self {
            name: name.to_string(),
            version: version.to_string(),
            weights,
            bias,
        }
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate().take(INPUT_DIM) {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    fn predict_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|inp| self.predict(inp)).collect()
    }
}

/// Simulated HTTP request
struct Request {
    method: &'static str,
    path: String,
    body: Option<String>,
}

/// Simulated HTTP response
struct Response {
    status: u16,
    body: String,
    latency_us: u64,
}

/// Server metrics collector
struct Metrics {
    requests_total: u64,
    requests_by_endpoint: HashMap<String, u64>,
    errors_total: u64,
    latency_sum_us: u64,
    latency_count: u64,
    predictions_total: u64,
}

impl Metrics {
    fn new() -> Self {
        Self {
            requests_total: 0,
            requests_by_endpoint: HashMap::new(),
            errors_total: 0,
            latency_sum_us: 0,
            latency_count: 0,
            predictions_total: 0,
        }
    }

    fn record_request(&mut self, path: &str, latency_us: u64, is_error: bool) {
        self.requests_total += 1;
        *self
            .requests_by_endpoint
            .entry(path.to_string())
            .or_insert(0) += 1;
        self.latency_sum_us += latency_us;
        self.latency_count += 1;
        if is_error {
            self.errors_total += 1;
        }
    }

    fn record_predictions(&mut self, count: u64) {
        self.predictions_total += count;
    }

    fn avg_latency_us(&self) -> f64 {
        if self.latency_count == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.latency_count as f64
        }
    }

    fn format_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# HELP requests_total Total HTTP requests\nrequests_total {}\n",
            self.requests_total
        ));
        out.push_str(&format!(
            "# HELP errors_total Total error responses\nerrors_total {}\n",
            self.errors_total
        ));
        out.push_str(&format!(
            "# HELP predictions_total Total predictions served\npredictions_total {}\n",
            self.predictions_total
        ));
        out.push_str(&format!(
            "# HELP latency_avg_us Average request latency\nlatency_avg_us {:.1}\n",
            self.avg_latency_us()
        ));
        for (endpoint, count) in &self.requests_by_endpoint {
            out.push_str(&format!(
                "requests_by_endpoint{{path=\"{}\"}} {}\n",
                endpoint, count
            ));
        }
        out
    }
}

/// Model server that handles routing and dispatch
struct ModelServer {
    models: HashMap<String, ServedModel>,
    metrics: Metrics,
    started_at: Instant,
}

impl ModelServer {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            metrics: Metrics::new(),
            started_at: Instant::now(),
        }
    }

    fn load_model(&mut self, name: &str, version: &str, seed: u64) {
        let model = ServedModel::new(name, version, seed);
        self.models.insert(name.to_string(), model);
    }

    fn handle_request(&mut self, request: &Request) -> Response {
        let start = Instant::now();
        let response = match (request.method, request.path.as_str()) {
            ("GET", "/health") => self.handle_health(),
            ("GET", "/metrics") => self.handle_metrics(),
            ("GET", "/v1/models") => self.handle_list_models(),
            ("POST", "/v1/predict") => self.handle_predict(request),
            ("POST", "/v1/batch") => self.handle_batch(request),
            _ => Response {
                status: 404,
                body: format!("{{\"error\":\"Not found: {}\"}}", request.path),
                latency_us: 0,
            },
        };

        let latency_us = start.elapsed().as_micros() as u64;
        let is_error = response.status >= 400;
        self.metrics
            .record_request(&request.path, latency_us, is_error);

        Response {
            latency_us,
            ..response
        }
    }

    fn handle_health(&self) -> Response {
        let uptime_ms = self.started_at.elapsed().as_millis();
        Response {
            status: 200,
            body: format!(
                "{{\"status\":\"healthy\",\"models\":{},\"uptime_ms\":{}}}",
                self.models.len(),
                uptime_ms
            ),
            latency_us: 0,
        }
    }

    fn handle_metrics(&self) -> Response {
        Response {
            status: 200,
            body: self.metrics.format_prometheus(),
            latency_us: 0,
        }
    }

    fn handle_list_models(&self) -> Response {
        let models: Vec<String> = self
            .models
            .values()
            .map(|m| format!("{{\"name\":\"{}\",\"version\":\"{}\"}}", m.name, m.version))
            .collect();
        Response {
            status: 200,
            body: format!("{{\"models\":[{}]}}", models.join(",")),
            latency_us: 0,
        }
    }

    fn handle_predict(&mut self, request: &Request) -> Response {
        let Some(body) = &request.body else {
            return Response {
                status: 400,
                body: "{\"error\":\"Missing request body\"}".to_string(),
                latency_us: 0,
            };
        };

        // Parse model name and input from simulated JSON
        let (model_name, input) = parse_predict_request(body);

        let Some(model) = self.models.get(&model_name) else {
            return Response {
                status: 404,
                body: format!("{{\"error\":\"Model '{}' not found\"}}", model_name),
                latency_us: 0,
            };
        };

        let output = model.predict(&input);
        self.metrics.record_predictions(1);

        Response {
            status: 200,
            body: format!("{{\"model\":\"{}\",\"output\":{:?}}}", model_name, output),
            latency_us: 0,
        }
    }

    fn handle_batch(&mut self, request: &Request) -> Response {
        let Some(body) = &request.body else {
            return Response {
                status: 400,
                body: "{\"error\":\"Missing request body\"}".to_string(),
                latency_us: 0,
            };
        };

        let (model_name, inputs) = parse_batch_request(body);

        let Some(model) = self.models.get(&model_name) else {
            return Response {
                status: 404,
                body: format!("{{\"error\":\"Model '{}' not found\"}}", model_name),
                latency_us: 0,
            };
        };

        let batch_size = inputs.len();
        let outputs = model.predict_batch(&inputs);
        self.metrics.record_predictions(batch_size as u64);

        Response {
            status: 200,
            body: format!(
                "{{\"model\":\"{}\",\"batch_size\":{},\"outputs_count\":{}}}",
                model_name,
                batch_size,
                outputs.len()
            ),
            latency_us: 0,
        }
    }
}

/// Parse a predict request body (simulated JSON parsing)
fn parse_predict_request(body: &str) -> (String, Vec<f32>) {
    // In production, use serde_json. Here we use deterministic mock.
    let model_name = if body.contains("classifier") {
        "classifier"
    } else {
        "default"
    };

    let mut hasher = DefaultHasher::new();
    body.hash(&mut hasher);
    let seed = hasher.finish();

    let input: Vec<f32> = (0..INPUT_DIM)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            h.finish() as f32 / u64::MAX as f32 - 0.5
        })
        .collect();

    (model_name.to_string(), input)
}

/// Parse a batch request body
fn parse_batch_request(body: &str) -> (String, Vec<Vec<f32>>) {
    let model_name = if body.contains("classifier") {
        "classifier"
    } else {
        "default"
    };

    let mut hasher = DefaultHasher::new();
    body.hash(&mut hasher);
    let seed = hasher.finish();

    // Extract batch size from body (simplified)
    let batch_size = body.matches("input").count().clamp(1, 32);

    let inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|b| {
            (0..INPUT_DIM)
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    (seed, b, i).hash(&mut h);
                    h.finish() as f32 / u64::MAX as f32 - 0.5
                })
                .collect()
        })
        .collect();

    (model_name.to_string(), inputs)
}

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
