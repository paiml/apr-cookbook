#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub const INPUT_DIM: usize = 32;
pub const OUTPUT_DIM: usize = 8;

/// Model weights for serving
pub struct ServedModel {
    pub name: String,
    pub version: String,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ServedModel {
    pub fn new(name: &str, version: &str, seed: u64) -> Self {
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

    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate().take(INPUT_DIM) {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|inp| self.predict(inp)).collect()
    }
}

/// Simulated HTTP request
pub struct Request {
    pub method: &'static str,
    pub path: String,
    pub body: Option<String>,
}

/// Simulated HTTP response
pub struct Response {
    pub status: u16,
    pub body: String,
    pub latency_us: u64,
}

/// Server metrics collector
pub struct Metrics {
    pub requests_total: u64,
    pub requests_by_endpoint: HashMap<String, u64>,
    pub errors_total: u64,
    pub latency_sum_us: u64,
    pub latency_count: u64,
    pub predictions_total: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total: 0,
            requests_by_endpoint: HashMap::new(),
            errors_total: 0,
            latency_sum_us: 0,
            latency_count: 0,
            predictions_total: 0,
        }
    }

    pub fn record_request(&mut self, path: &str, latency_us: u64, is_error: bool) {
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

    pub fn record_predictions(&mut self, count: u64) {
        self.predictions_total += count;
    }

    pub fn avg_latency_us(&self) -> f64 {
        if self.latency_count == 0 {
            0.0
        } else {
            self.latency_sum_us as f64 / self.latency_count as f64
        }
    }

    pub fn format_prometheus(&self) -> String {
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
pub struct ModelServer {
    pub models: HashMap<String, ServedModel>,
    pub metrics: Metrics,
    pub started_at: Instant,
}

impl ModelServer {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            metrics: Metrics::new(),
            started_at: Instant::now(),
        }
    }

    pub fn load_model(&mut self, name: &str, version: &str, seed: u64) {
        let model = ServedModel::new(name, version, seed);
        self.models.insert(name.to_string(), model);
    }

    pub fn handle_request(&mut self, request: &Request) -> Response {
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

    pub fn handle_health(&self) -> Response {
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

    pub fn handle_metrics(&self) -> Response {
        Response {
            status: 200,
            body: self.metrics.format_prometheus(),
            latency_us: 0,
        }
    }

    pub fn handle_list_models(&self) -> Response {
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

    pub fn handle_predict(&mut self, request: &Request) -> Response {
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

    pub fn handle_batch(&mut self, request: &Request) -> Response {
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
pub fn parse_predict_request(body: &str) -> (String, Vec<f32>) {
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
pub fn parse_batch_request(body: &str) -> (String, Vec<Vec<f32>>) {
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
