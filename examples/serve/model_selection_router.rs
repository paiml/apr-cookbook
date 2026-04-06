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

const NUM_VARIANTS: usize = 3;
const VARIANT_NAMES: [&str; NUM_VARIANTS] = ["fast-small", "balanced", "accurate-large"];

#[derive(Clone, Debug)]
struct ModelVariant {
    name: String,
    avg_latency_ms: f64,
    accuracy: f64,
    cost_per_request: f64,
}

impl ModelVariant {
    fn new(name: &str, avg_latency_ms: f64, accuracy: f64, cost_per_request: f64) -> Self {
        Self {
            name: name.to_string(),
            avg_latency_ms,
            accuracy,
            cost_per_request,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutingStrategy {
    RoundRobin,
    LowestLatency,
    CostAware,
    AccuracyWeighted,
    Adaptive,
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::LowestLatency => write!(f, "LowestLatency"),
            Self::CostAware => write!(f, "CostAware"),
            Self::AccuracyWeighted => write!(f, "AccuracyWeighted"),
            Self::Adaptive => write!(f, "Adaptive"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Priority {
    High,
    Medium,
    Low,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "High"),
            Self::Medium => write!(f, "Medium"),
            Self::Low => write!(f, "Low"),
        }
    }
}

#[derive(Clone, Debug)]
struct InferenceRequest {
    id: u64,
    priority: Priority,
    max_latency_ms: f64,
    max_cost: f64,
}

impl InferenceRequest {
    fn new(id: u64, priority: Priority, max_latency_ms: f64, max_cost: f64) -> Self {
        Self {
            id,
            priority,
            max_latency_ms,
            max_cost,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct RoutingDecision {
    request_id: u64,
    selected_model: String,
    strategy_used: RoutingStrategy,
    shadow_model: Option<String>,
}

#[derive(Clone, Debug)]
struct PerformanceObservation {
    model_name: String,
    latency_ms: f64,
    was_accurate: bool,
    cost: f64,
}

#[derive(Clone, Debug)]
struct RoutingMetrics {
    total_requests: usize,
    avg_latency: f64,
    avg_cost: f64,
    accuracy_distribution: HashMap<String, (usize, usize)>,
}

impl RoutingMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            avg_latency: 0.0,
            avg_cost: 0.0,
            accuracy_distribution: HashMap::new(),
        }
    }

    fn record(&mut self, obs: &PerformanceObservation) {
        self.total_requests += 1;
        let n = self.total_requests as f64;
        self.avg_latency += (obs.latency_ms - self.avg_latency) / n;
        self.avg_cost += (obs.cost - self.avg_cost) / n;
        let entry = self
            .accuracy_distribution
            .entry(obs.model_name.clone())
            .or_insert((0, 0));
        if obs.was_accurate {
            entry.0 += 1;
        }
        entry.1 += 1;
    }

    fn model_accuracy(&self, model_name: &str) -> f64 {
        self.accuracy_distribution
            .get(model_name)
            .map_or(
                0.0,
                |&(c, t)| if t == 0 { 0.0 } else { c as f64 / t as f64 },
            )
    }
}

struct ModelRouter {
    variants: Vec<ModelVariant>,
    strategy: RoutingStrategy,
    decisions: Vec<RoutingDecision>,
    metrics: RoutingMetrics,
    round_robin_counter: usize,
    adaptive_weights: Vec<f64>,
}

impl ModelRouter {
    fn new(variants: Vec<ModelVariant>, strategy: RoutingStrategy) -> Self {
        let n = variants.len();
        let w = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        Self {
            variants,
            strategy,
            decisions: Vec::new(),
            metrics: RoutingMetrics::new(),
            round_robin_counter: 0,
            adaptive_weights: vec![w; n],
        }
    }

    fn route(&mut self, request: &InferenceRequest) -> usize {
        let idx = match self.strategy {
            RoutingStrategy::RoundRobin => self.route_round_robin(),
            RoutingStrategy::LowestLatency => self.route_lowest_latency(request),
            RoutingStrategy::CostAware => self.route_cost_aware(request),
            RoutingStrategy::AccuracyWeighted => self.route_accuracy_weighted(request),
            RoutingStrategy::Adaptive => self.route_adaptive(request),
        };
        let shadow = self.pick_shadow(idx, request.id);
        self.decisions.push(RoutingDecision {
            request_id: request.id,
            selected_model: self.variants[idx].name.clone(),
            strategy_used: self.strategy,
            shadow_model: shadow.map(|s| self.variants[s].name.clone()),
        });
        idx
    }

    fn route_round_robin(&mut self) -> usize {
        let idx = self.round_robin_counter % self.variants.len();
        self.round_robin_counter += 1;
        idx
    }

    fn route_lowest_latency(&self, request: &InferenceRequest) -> usize {
        self.variants
            .iter()
            .enumerate()
            .filter(|(_, v)| v.avg_latency_ms <= request.max_latency_ms)
            .min_by(|(_, a), (_, b)| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i)
    }

    fn route_cost_aware(&self, request: &InferenceRequest) -> usize {
        self.variants
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v.cost_per_request <= request.max_cost && v.avg_latency_ms <= request.max_latency_ms
            })
            .min_by(|(_, a), (_, b)| {
                a.cost_per_request
                    .partial_cmp(&b.cost_per_request)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(
                || {
                    self.variants
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.cost_per_request
                                .partial_cmp(&b.cost_per_request)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or(0, |(i, _)| i)
                },
                |(i, _)| i,
            )
    }

    fn route_accuracy_weighted(&self, request: &InferenceRequest) -> usize {
        match request.priority {
            Priority::High => self
                .variants
                .iter()
                .enumerate()
                .filter(|(_, v)| v.avg_latency_ms <= request.max_latency_ms)
                .max_by(|(_, a), (_, b)| {
                    a.accuracy
                        .partial_cmp(&b.accuracy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or_else(
                    || {
                        self.variants
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.accuracy
                                    .partial_cmp(&b.accuracy)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map_or(0, |(i, _)| i)
                    },
                    |(i, _)| i,
                ),
            Priority::Medium => {
                let max_cost = self
                    .variants
                    .iter()
                    .map(|v| v.cost_per_request)
                    .fold(f64::NEG_INFINITY, f64::max);
                let max_lat = self
                    .variants
                    .iter()
                    .map(|v| v.avg_latency_ms)
                    .fold(f64::NEG_INFINITY, f64::max);
                self.variants
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        Self::balanced_score(a, max_cost, max_lat)
                            .partial_cmp(&Self::balanced_score(b, max_cost, max_lat))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i)
            }
            Priority::Low => self.route_cost_aware(request),
        }
    }

    fn balanced_score(v: &ModelVariant, max_cost: f64, max_lat: f64) -> f64 {
        let nc = if max_cost > 0.0 {
            v.cost_per_request / max_cost
        } else {
            0.0
        };
        let nl = if max_lat > 0.0 {
            v.avg_latency_ms / max_lat
        } else {
            0.0
        };
        v.accuracy * 0.5 + (1.0 - nc) * 0.3 + (1.0 - nl) * 0.2
    }

    fn route_adaptive(&self, request: &InferenceRequest) -> usize {
        let mut h = DefaultHasher::new();
        request.id.hash(&mut h);
        let hash_val = h.finish() as f64 / u64::MAX as f64;
        let mut cumulative = 0.0;
        for (i, &w) in self.adaptive_weights.iter().enumerate() {
            cumulative += w;
            if hash_val < cumulative {
                return i;
            }
        }
        self.variants.len() - 1
    }

    fn pick_shadow(&self, primary_idx: usize, request_id: u64) -> Option<usize> {
        let mut h = DefaultHasher::new();
        ("shadow", request_id).hash(&mut h);
        if (h.finish() % 2) != 0 {
            return None;
        }
        let best = self
            .variants
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.accuracy
                    .partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);
        if best == primary_idx {
            None
        } else {
            Some(best)
        }
    }

    fn record_observation(&mut self, obs: &PerformanceObservation) {
        self.metrics.record(obs);
    }

    fn update_adaptive_weights(&mut self) {
        if self.variants.is_empty() {
            return;
        }
        let scores: Vec<f64> = self
            .variants
            .iter()
            .map(|v| {
                let acc = {
                    let o = self.metrics.model_accuracy(&v.name);
                    if o > 0.0 {
                        o
                    } else {
                        v.accuracy
                    }
                };
                acc / (v.avg_latency_ms.max(1.0) * v.cost_per_request.max(0.001))
            })
            .collect();
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            self.adaptive_weights = scores.iter().map(|&s| s / total).collect();
        }
    }

    fn decision_count(&self) -> usize {
        self.decisions.len()
    }
    fn shadow_count(&self) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.shadow_model.is_some())
            .count()
    }
    fn decisions_for_model(&self, name: &str) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.selected_model == name)
            .count()
    }
}

fn simulate_latency(v: &ModelVariant, id: u64) -> f64 {
    let mut h = DefaultHasher::new();
    ("latency", v.name.as_str(), id).hash(&mut h);
    (v.avg_latency_ms + (h.finish() % 20) as f64 - 10.0).max(1.0)
}

fn simulate_accuracy(v: &ModelVariant, id: u64) -> bool {
    let mut h = DefaultHasher::new();
    ("accuracy", v.name.as_str(), id).hash(&mut h);
    (h.finish() as f64 / u64::MAX as f64) < v.accuracy
}

fn make_default_variants() -> Vec<ModelVariant> {
    vec![
        ModelVariant::new(VARIANT_NAMES[0], 10.0, 0.82, 0.001),
        ModelVariant::new(VARIANT_NAMES[1], 50.0, 0.91, 0.005),
        ModelVariant::new(VARIANT_NAMES[2], 200.0, 0.97, 0.020),
    ]
}

fn route_and_observe(router: &mut ModelRouter, start_id: u64, count: u64) {
    for i in 0..count {
        let req = InferenceRequest::new(start_id + i, Priority::Medium, 500.0, 1.0);
        let idx = router.route(&req);
        let v = &router.variants[idx];
        router.record_observation(&PerformanceObservation {
            model_name: v.name.clone(),
            latency_ms: simulate_latency(v, req.id),
            was_accurate: simulate_accuracy(v, req.id),
            cost: v.cost_per_request,
        });
    }
}

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
