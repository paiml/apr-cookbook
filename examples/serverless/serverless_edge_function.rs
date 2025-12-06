//! # Recipe: Edge Function Deployment
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
//! Deploy model at edge locations for low latency inference.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serverless_edge_function
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Get predefined edge locations
fn get_edge_locations() -> Vec<EdgeLocation> {
    vec![
        EdgeLocation {
            id: "us-east-1",
            name: "US East (Virginia)",
            latency_base_ms: 5,
        },
        EdgeLocation {
            id: "us-west-2",
            name: "US West (Oregon)",
            latency_base_ms: 8,
        },
        EdgeLocation {
            id: "eu-west-1",
            name: "EU (Ireland)",
            latency_base_ms: 12,
        },
        EdgeLocation {
            id: "ap-northeast-1",
            name: "Asia (Tokyo)",
            latency_base_ms: 15,
        },
        EdgeLocation {
            id: "ap-southeast-1",
            name: "Asia (Singapore)",
            latency_base_ms: 18,
        },
    ]
}

/// Get test client requests
fn get_test_requests() -> Vec<(&'static str, &'static str)> {
    vec![
        ("client-nyc", "us-east-1"),
        ("client-la", "us-west-2"),
        ("client-london", "eu-west-1"),
        ("client-tokyo", "ap-northeast-1"),
        ("client-singapore", "ap-southeast-1"),
    ]
}

/// Print request routing results
fn print_routing_results(deployment: &EdgeDeployment, requests: &[(&str, &str)]) -> Result<()> {
    println!("Request routing:");
    println!("{:-<60}", "");
    println!(
        "{:<20} {:<15} {:>10} {:>10}",
        "Client", "Edge", "Latency", "Status"
    );
    println!("{:-<60}", "");

    for (client, region) in requests {
        let result = deployment.route_request(client, region)?;
        println!(
            "{:<20} {:<15} {:>8}ms {:>10}",
            client, result.edge_location, result.latency_ms, result.status
        );
    }
    println!("{:-<60}", "");
    Ok(())
}

/// Calculate and print latency comparison
fn print_latency_comparison(deployment: &EdgeDeployment, requests: &[(&str, &str)]) -> (f64, f64) {
    let total_edge: u32 = requests
        .iter()
        .map(|(_, region)| deployment.get_edge_latency(region))
        .sum();
    let total_central = 50u32 * requests.len() as u32;

    let avg_edge = f64::from(total_edge) / requests.len() as f64;
    let avg_central = f64::from(total_central) / requests.len() as f64;
    let improvement = ((avg_central - avg_edge) / avg_central) * 100.0;

    println!();
    println!("Latency comparison (Edge vs Centralized):");
    println!("  Average edge latency: {:.1}ms", avg_edge);
    println!("  Average central latency: {:.1}ms", avg_central);
    println!("  Improvement: {:.1}%", improvement);

    (avg_edge, improvement)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("serverless_edge_function")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Edge function deployment simulation");
    println!();

    let locations = get_edge_locations();
    ctx.record_metric("edge_locations", locations.len() as i64);

    let mut deployment = EdgeDeployment::new("fraud-detector-edge");
    println!("Deploying to edge locations:");
    for loc in &locations {
        deployment.deploy(loc)?;
        println!("  ✓ {}: {}", loc.id, loc.name);
    }
    println!();

    let requests = get_test_requests();
    print_routing_results(&deployment, &requests)?;

    let (avg_edge, improvement) = print_latency_comparison(&deployment, &requests);
    ctx.record_float_metric("avg_edge_latency_ms", avg_edge);
    ctx.record_float_metric("latency_improvement_pct", improvement);

    let config_path = ctx.path("edge_deployment.json");
    deployment.save(&config_path)?;
    println!();
    println!("Deployment config saved to: {:?}", config_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeLocation {
    id: &'static str,
    name: &'static str,
    latency_base_ms: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct EdgeDeployment {
    function_name: String,
    locations: HashMap<String, EdgeLocationState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeLocationState {
    id: String,
    name: String,
    status: String,
    latency_ms: u32,
}

#[derive(Debug)]
struct RouteResult {
    edge_location: String,
    latency_ms: u32,
    status: String,
}

impl EdgeDeployment {
    fn new(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            locations: HashMap::new(),
        }
    }

    fn deploy(&mut self, location: &EdgeLocation) -> Result<()> {
        self.locations.insert(
            location.id.to_string(),
            EdgeLocationState {
                id: location.id.to_string(),
                name: location.name.to_string(),
                status: "active".to_string(),
                latency_ms: location.latency_base_ms,
            },
        );
        Ok(())
    }

    fn route_request(&self, _client: &str, region: &str) -> Result<RouteResult> {
        let location = self
            .locations
            .get(region)
            .ok_or_else(|| CookbookError::ModelNotFound {
                path: std::path::PathBuf::from(region),
            })?;

        Ok(RouteResult {
            edge_location: location.id.clone(),
            latency_ms: location.latency_ms,
            status: "success".to_string(),
        })
    }

    fn get_edge_latency(&self, region: &str) -> u32 {
        self.locations.get(region).map_or(50, |l| l.latency_ms)
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_creation() {
        let deployment = EdgeDeployment::new("test-function");
        assert_eq!(deployment.function_name, "test-function");
        assert!(deployment.locations.is_empty());
    }

    #[test]
    fn test_deploy_location() {
        let mut deployment = EdgeDeployment::new("test");
        let location = EdgeLocation {
            id: "us-east-1",
            name: "US East",
            latency_base_ms: 5,
        };

        deployment.deploy(&location).unwrap();

        assert!(deployment.locations.contains_key("us-east-1"));
    }

    #[test]
    fn test_route_request() {
        let mut deployment = EdgeDeployment::new("test");
        deployment
            .deploy(&EdgeLocation {
                id: "us-east-1",
                name: "US East",
                latency_base_ms: 10,
            })
            .unwrap();

        let result = deployment.route_request("client", "us-east-1").unwrap();

        assert_eq!(result.edge_location, "us-east-1");
        assert_eq!(result.latency_ms, 10);
    }

    #[test]
    fn test_route_unknown_region() {
        let deployment = EdgeDeployment::new("test");
        let result = deployment.route_request("client", "unknown");

        assert!(result.is_err());
    }

    #[test]
    fn test_save_deployment() {
        let ctx = RecipeContext::new("test_edge_save").unwrap();
        let path = ctx.path("deployment.json");

        let deployment = EdgeDeployment::new("test");
        deployment.save(&path).unwrap();

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
        fn prop_deploy_adds_location(n in 1usize..10) {
            let mut deployment = EdgeDeployment::new("test");

            for i in 0..n {
                // We need to use owned strings here
                let id = format!("region-{}", i);
                let name = format!("Region {}", i);

                deployment.locations.insert(
                    id.clone(),
                    EdgeLocationState {
                        id,
                        name,
                        status: "active".to_string(),
                        latency_ms: 10,
                    },
                );
            }

            prop_assert_eq!(deployment.locations.len(), n);
        }

        #[test]
        fn prop_latency_positive(latency in 1u32..100) {
            let mut deployment = EdgeDeployment::new("test");
            deployment.locations.insert(
                "test".to_string(),
                EdgeLocationState {
                    id: "test".to_string(),
                    name: "Test".to_string(),
                    status: "active".to_string(),
                    latency_ms: latency,
                },
            );

            let result = deployment.route_request("client", "test").unwrap();
            prop_assert!(result.latency_ms > 0);
        }
    }
}
