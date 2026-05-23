#![allow(unused_imports)]
//! Model Selection Router
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Intelligent request routing to multiple model variants based on latency,
//! accuracy, and cost tradeoffs. Supports round-robin, lowest-latency,
//! cost-aware, accuracy-weighted, and adaptive routing with shadow traffic.
//!
//! ```bash
//! cargo run --example model_selection_router
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Model Selection Router Example ===\n");
    let variants = make_default_variants();

    // 1. Registry
    println!("1. Model Variant Registry");
    for v in &variants {
        println!(
            "   {:>16} lat={:.0}ms acc={:.0}% cost=${:.3}",
            v.name,
            v.avg_latency_ms,
            v.accuracy * 100.0,
            v.cost_per_request
        );
    }

    // 2. Round-robin baseline
    println!("\n2. Round-Robin Baseline (30 requests)");
    let mut rr = ModelRouter::new(variants.clone(), RoutingStrategy::RoundRobin);
    for i in 0..30u64 {
        let req = InferenceRequest::new(i, Priority::Medium, 500.0, 1.0);
        let idx = rr.route(&req);
        let v = &rr.variants[idx];
        rr.record_observation(&PerformanceObservation {
            model_name: v.name.clone(),
            latency_ms: simulate_latency(v, i),
            was_accurate: simulate_accuracy(v, i),
            cost: v.cost_per_request,
        });
    }
    for n in VARIANT_NAMES {
        println!("   {n}: {} requests", rr.decisions_for_model(n));
    }
    println!(
        "   Avg latency: {:.1}ms, cost: ${:.4}",
        rr.metrics.avg_latency, rr.metrics.avg_cost
    );

    // 3. Latency-optimized
    println!("\n3. Latency-Optimized Routing");
    let mut lr = ModelRouter::new(variants.clone(), RoutingStrategy::LowestLatency);
    for (id, lat) in [(100u64, 20.0), (101, 100.0), (102, 500.0)] {
        let req = InferenceRequest::new(id, Priority::Medium, lat, 1.0);
        let idx = lr.route(&req);
        println!("   id={id} maxLat={lat:.0}ms -> {}", lr.variants[idx].name);
    }

    // 4. Cost-aware + accuracy-weighted
    println!("\n4. Cost-Aware Routing");
    let mut cr = ModelRouter::new(variants.clone(), RoutingStrategy::CostAware);
    for (id, cost, lat) in [
        (200u64, 0.002, 500.0),
        (201, 0.010, 500.0),
        (202, 0.050, 500.0),
    ] {
        let req = InferenceRequest::new(id, Priority::Medium, lat, cost);
        let idx = cr.route(&req);
        println!(
            "   id={id} budget=${cost:.3} -> {} (${:.3})",
            cr.variants[idx].name, cr.variants[idx].cost_per_request
        );
    }
    println!("\n   Accuracy-Weighted by Priority:");
    let mut ar = ModelRouter::new(variants.clone(), RoutingStrategy::AccuracyWeighted);
    for (id, pri) in [
        (250u64, Priority::High),
        (251, Priority::Medium),
        (252, Priority::Low),
    ] {
        let req = InferenceRequest::new(id, pri, 500.0, 1.0);
        let idx = ar.route(&req);
        println!("   id={id} {pri} -> {}", ar.variants[idx].name);
    }

    // 5. Shadow traffic
    println!("\n5. Shadow Traffic (20 requests)");
    let mut sr = ModelRouter::new(variants.clone(), RoutingStrategy::LowestLatency);
    for i in 300..320u64 {
        let req = InferenceRequest::new(i, Priority::Medium, 100.0, 1.0);
        let idx = sr.route(&req);
        let v = &sr.variants[idx];
        sr.record_observation(&PerformanceObservation {
            model_name: v.name.clone(),
            latency_ms: simulate_latency(v, i),
            was_accurate: simulate_accuracy(v, i),
            cost: v.cost_per_request,
        });
    }
    println!(
        "   Shadow rate: {}/{} ({:.0}%)",
        sr.shadow_count(),
        20,
        sr.shadow_count() as f64 / 20.0 * 100.0
    );

    // 6. Adaptive routing
    println!("\n6. Adaptive Routing with Feedback");
    let mut ad = ModelRouter::new(variants.clone(), RoutingStrategy::Adaptive);
    route_and_observe(&mut ad, 400, 50);
    ad.update_adaptive_weights();
    println!("   Weights after 50 requests:");
    for (i, w) in ad.adaptive_weights.iter().enumerate() {
        println!("     {}: {:.3}", ad.variants[i].name, w);
    }
    route_and_observe(&mut ad, 450, 50);
    println!(
        "   Total: {} decisions, {} shadow\n",
        ad.decision_count(),
        ad.shadow_count()
    );
    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_cycles_evenly() {
        let mut router = ModelRouter::new(make_default_variants(), RoutingStrategy::RoundRobin);
        let mut counts = [0usize; NUM_VARIANTS];
        for i in 0..30u64 {
            counts[router.route(&InferenceRequest::new(i, Priority::Medium, 500.0, 1.0))] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, 10, "Model {i} got {c} instead of 10");
        }
    }

    #[test]
    fn test_lowest_latency_picks_fastest() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::LowestLatency);
        assert_eq!(
            r.route(&InferenceRequest::new(1, Priority::Medium, 500.0, 1.0)),
            0
        );
    }

    #[test]
    fn test_lowest_latency_sla_fallback() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::LowestLatency);
        assert_eq!(
            r.route(&InferenceRequest::new(1, Priority::High, 5.0, 1.0)),
            0
        );
    }

    #[test]
    fn test_cost_aware_picks_cheapest() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::CostAware);
        let idx = r.route(&InferenceRequest::new(1, Priority::Medium, 500.0, 1.0));
        assert_eq!(r.variants[idx].name, "fast-small");
    }

    #[test]
    fn test_accuracy_weighted_high_priority() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::AccuracyWeighted);
        let idx = r.route(&InferenceRequest::new(1, Priority::High, 500.0, 1.0));
        assert_eq!(r.variants[idx].name, "accurate-large");
    }

    #[test]
    fn test_accuracy_weighted_low_priority() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::AccuracyWeighted);
        let idx = r.route(&InferenceRequest::new(1, Priority::Low, 500.0, 1.0));
        assert_eq!(r.variants[idx].name, "fast-small");
    }

    #[test]
    fn test_adaptive_weights_update() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::Adaptive);
        for _ in 0..20u64 {
            r.record_observation(&PerformanceObservation {
                model_name: "accurate-large".to_string(),
                latency_ms: 200.0,
                was_accurate: true,
                cost: 0.020,
            });
            r.record_observation(&PerformanceObservation {
                model_name: "fast-small".to_string(),
                latency_ms: 10.0,
                was_accurate: false,
                cost: 0.001,
            });
        }
        r.update_adaptive_weights();
        let sum: f64 = r.adaptive_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(r.adaptive_weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_shadow_traffic_deterministic() {
        let variants = make_default_variants();
        let mut r1 = ModelRouter::new(variants.clone(), RoutingStrategy::RoundRobin);
        let mut r2 = ModelRouter::new(variants, RoutingStrategy::RoundRobin);
        for i in 0..20u64 {
            let req = InferenceRequest::new(i, Priority::Medium, 500.0, 1.0);
            r1.route(&req);
            r2.route(&req);
        }
        assert_eq!(r1.shadow_count(), r2.shadow_count());
    }

    #[test]
    fn test_metrics_running_average() {
        let mut m = RoutingMetrics::new();
        m.record(&PerformanceObservation {
            model_name: "m1".to_string(),
            latency_ms: 10.0,
            was_accurate: true,
            cost: 0.01,
        });
        m.record(&PerformanceObservation {
            model_name: "m1".to_string(),
            latency_ms: 20.0,
            was_accurate: false,
            cost: 0.03,
        });
        assert!((m.avg_latency - 15.0).abs() < 1e-10);
        assert!((m.avg_cost - 0.02).abs() < 1e-10);
        assert!((m.model_accuracy("m1") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_audit_trail() {
        let mut r = ModelRouter::new(make_default_variants(), RoutingStrategy::RoundRobin);
        for i in 0..15u64 {
            r.route(&InferenceRequest::new(i, Priority::Medium, 500.0, 1.0));
        }
        assert_eq!(r.decision_count(), 15);
        for (i, d) in r.decisions.iter().enumerate() {
            assert_eq!(d.request_id, i as u64);
        }
    }
}
