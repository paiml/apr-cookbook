//! # Recipe: Model Health Check API
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
//! Health check endpoint for deployed model monitoring.
//!
//! ## Run Command
//! ```bash
//! cargo run --example api_model_health_check
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("api_model_health_check")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Model health check endpoint");
    println!();

    // Create mock model endpoints
    let endpoints = vec![
        ModelEndpoint {
            name: "fraud-detector".to_string(),
            url: "http://localhost:8080/v1/fraud".to_string(),
            version: "1.2.0".to_string(),
        },
        ModelEndpoint {
            name: "sentiment-analyzer".to_string(),
            url: "http://localhost:8081/v1/sentiment".to_string(),
            version: "2.0.1".to_string(),
        },
        ModelEndpoint {
            name: "image-classifier".to_string(),
            url: "http://localhost:8082/v1/classify".to_string(),
            version: "1.0.0".to_string(),
        },
    ];

    ctx.record_metric("endpoints", endpoints.len() as i64);

    // Run health checks
    println!("Running health checks...");
    println!();

    let mut health_results = Vec::new();
    for endpoint in &endpoints {
        let result = check_health(endpoint);
        health_results.push(result);
    }

    // Display results
    println!("{:-<70}", "");
    println!(
        "{:<20} {:<10} {:<15} {:>10} {:>10}",
        "Model", "Status", "Version", "Latency", "Memory"
    );
    println!("{:-<70}", "");

    let mut healthy_count = 0;
    for result in &health_results {
        let status_str = if result.healthy {
            "HEALTHY"
        } else {
            "UNHEALTHY"
        };
        if result.healthy {
            healthy_count += 1;
        }

        println!(
            "{:<20} {:<10} {:<15} {:>8}ms {:>8}MB",
            result.name, status_str, result.version, result.latency_ms, result.memory_mb
        );
    }
    println!("{:-<70}", "");

    ctx.record_metric("healthy", i64::from(healthy_count));
    ctx.record_metric(
        "unhealthy",
        health_results.len() as i64 - i64::from(healthy_count),
    );

    // Aggregate health
    let aggregate = aggregate_health(&health_results);
    println!();
    println!("Aggregate Health:");
    println!(
        "  Status: {}",
        if aggregate.all_healthy {
            "ALL HEALTHY"
        } else {
            "DEGRADED"
        }
    );
    println!(
        "  Healthy: {}/{}",
        aggregate.healthy_count, aggregate.total_count
    );
    println!("  Avg latency: {:.1}ms", aggregate.avg_latency_ms);
    println!("  Total memory: {}MB", aggregate.total_memory_mb);

    // Save health report
    let report_path = ctx.path("health_report.json");
    save_health_report(&report_path, &health_results, &aggregate)?;
    println!();
    println!("Health report saved to: {:?}", report_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelEndpoint {
    name: String,
    url: String,
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HealthResult {
    name: String,
    healthy: bool,
    version: String,
    latency_ms: u32,
    memory_mb: u32,
    checks: HashMap<String, bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AggregateHealth {
    all_healthy: bool,
    healthy_count: usize,
    total_count: usize,
    avg_latency_ms: f64,
    total_memory_mb: u32,
}

fn check_health(endpoint: &ModelEndpoint) -> HealthResult {
    // Deterministic mock health check based on endpoint name
    let seed = hash_name_to_seed(&endpoint.name);

    // Mock checks
    let mut checks = HashMap::new();
    checks.insert("model_loaded".to_string(), true);
    checks.insert("memory_ok".to_string(), true);
    checks.insert("inference_ok".to_string(), true);
    checks.insert("dependencies_ok".to_string(), true);

    // Deterministic latency and memory based on seed
    let latency = 10 + (seed % 50) as u32;
    let memory = 100 + (seed % 400) as u32;

    HealthResult {
        name: endpoint.name.clone(),
        healthy: checks.values().all(|&v| v),
        version: endpoint.version.clone(),
        latency_ms: latency,
        memory_mb: memory,
        checks,
    }
}

fn aggregate_health(results: &[HealthResult]) -> AggregateHealth {
    let healthy_count = results.iter().filter(|r| r.healthy).count();
    let total_latency: u32 = results.iter().map(|r| r.latency_ms).sum();
    let total_memory: u32 = results.iter().map(|r| r.memory_mb).sum();

    AggregateHealth {
        all_healthy: healthy_count == results.len(),
        healthy_count,
        total_count: results.len(),
        avg_latency_ms: if results.is_empty() {
            0.0
        } else {
            f64::from(total_latency) / results.len() as f64
        },
        total_memory_mb: total_memory,
    }
}

fn save_health_report(
    path: &std::path::Path,
    results: &[HealthResult],
    aggregate: &AggregateHealth,
) -> Result<()> {
    #[derive(Serialize)]
    struct Report<'a> {
        timestamp: u64,
        results: &'a [HealthResult],
        aggregate: &'a AggregateHealth,
    }

    let report = Report {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        results,
        aggregate,
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_check() {
        let endpoint = ModelEndpoint {
            name: "test-model".to_string(),
            url: "http://localhost".to_string(),
            version: "1.0.0".to_string(),
        };

        let result = check_health(&endpoint);

        assert!(result.healthy);
        assert_eq!(result.name, "test-model");
        assert_eq!(result.version, "1.0.0");
    }

    #[test]
    fn test_deterministic_health() {
        let endpoint = ModelEndpoint {
            name: "test".to_string(),
            url: "http://localhost".to_string(),
            version: "1.0.0".to_string(),
        };

        let r1 = check_health(&endpoint);
        let r2 = check_health(&endpoint);

        assert_eq!(r1.latency_ms, r2.latency_ms);
        assert_eq!(r1.memory_mb, r2.memory_mb);
    }

    #[test]
    fn test_aggregate_all_healthy() {
        let results = vec![
            HealthResult {
                name: "m1".to_string(),
                healthy: true,
                version: "1.0".to_string(),
                latency_ms: 10,
                memory_mb: 100,
                checks: HashMap::new(),
            },
            HealthResult {
                name: "m2".to_string(),
                healthy: true,
                version: "1.0".to_string(),
                latency_ms: 20,
                memory_mb: 200,
                checks: HashMap::new(),
            },
        ];

        let aggregate = aggregate_health(&results);

        assert!(aggregate.all_healthy);
        assert_eq!(aggregate.healthy_count, 2);
        assert_eq!(aggregate.total_count, 2);
        assert!((aggregate.avg_latency_ms - 15.0).abs() < 0.01);
        assert_eq!(aggregate.total_memory_mb, 300);
    }

    #[test]
    fn test_aggregate_partial_healthy() {
        let results = vec![
            HealthResult {
                name: "m1".to_string(),
                healthy: true,
                version: "1.0".to_string(),
                latency_ms: 10,
                memory_mb: 100,
                checks: HashMap::new(),
            },
            HealthResult {
                name: "m2".to_string(),
                healthy: false,
                version: "1.0".to_string(),
                latency_ms: 20,
                memory_mb: 200,
                checks: HashMap::new(),
            },
        ];

        let aggregate = aggregate_health(&results);

        assert!(!aggregate.all_healthy);
        assert_eq!(aggregate.healthy_count, 1);
    }

    #[test]
    fn test_save_report() {
        let ctx = RecipeContext::new("test_health_report").unwrap();
        let path = ctx.path("report.json");

        let results = vec![];
        let aggregate = aggregate_health(&results);

        save_health_report(&path, &results, &aggregate).unwrap();
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
        fn prop_aggregate_counts_match(n in 0usize..20) {
            let results: Vec<_> = (0..n)
                .map(|i| HealthResult {
                    name: format!("m{}", i),
                    healthy: true,
                    version: "1.0".to_string(),
                    latency_ms: 10,
                    memory_mb: 100,
                    checks: HashMap::new(),
                })
                .collect();

            let aggregate = aggregate_health(&results);
            prop_assert_eq!(aggregate.total_count, n);
            prop_assert_eq!(aggregate.healthy_count, n);
        }

        #[test]
        fn prop_total_memory_sums(memories in proptest::collection::vec(1u32..500, 1..10)) {
            let results: Vec<_> = memories
                .iter()
                .enumerate()
                .map(|(i, &mem)| HealthResult {
                    name: format!("m{}", i),
                    healthy: true,
                    version: "1.0".to_string(),
                    latency_ms: 10,
                    memory_mb: mem,
                    checks: HashMap::new(),
                })
                .collect();

            let aggregate = aggregate_health(&results);
            let expected: u32 = memories.iter().sum();
            prop_assert_eq!(aggregate.total_memory_mb, expected);
        }
    }
}
