//! Model Selection Router Example
//!
//! Demonstrates intelligent request routing to multiple model variants based
//! on latency, accuracy, and cost tradeoffs. Supports round-robin, lowest-latency,
//! cost-aware, accuracy-weighted, and adaptive routing strategies with shadow
//! traffic mirroring and a performance feedback loop.
//!
//! ```text
//! Routing Pipeline:
//!
//!   [Request] ──→ [Router] ──→ [Strategy Selection]
//!                    │             │
//!                    │         ┌───┴───────────────────────┐
//!                    │         │ RoundRobin                │
//!                    │         │ LowestLatency             │
//!                    │         │ CostAware (budget check)  │
//!                    │         │ AccuracyWeighted          │
//!                    │         │ Adaptive (feedback loop)  │
//!                    │         └───┬───────────────────────┘
//!                    │             │
//!                    │             ▼
//!                    │      [Primary Model] ──→ Response
//!                    │             │
//!                    │      [Shadow Model?] ──→ Compare (no effect on response)
//!                    │
//!                    └──→ [Audit Trail] ──→ [Feedback Loop]
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_selection_router
//! ```

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NUM_VARIANTS: usize = 3;
const VARIANT_NAMES: [&str; NUM_VARIANTS] = ["fast-small", "balanced", "accurate-large"];

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A model variant with known performance characteristics.
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

/// Strategy used to select which model handles a request.
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

/// Priority level for an inference request.
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

/// An incoming inference request with SLA metadata.
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

/// Record of a single routing decision for the audit trail.
#[derive(Clone, Debug)]
struct RoutingDecision {
    request_id: u64,
    selected_model: String,
    strategy_used: RoutingStrategy,
    shadow_model: Option<String>,
}

/// Observed performance for a single request handled by a model.
#[derive(Clone, Debug)]
struct PerformanceObservation {
    model_name: String,
    latency_ms: f64,
    was_accurate: bool,
    cost: f64,
}

/// Aggregated routing metrics.
#[derive(Clone, Debug)]
struct RoutingMetrics {
    total_requests: usize,
    avg_latency: f64,
    avg_cost: f64,
    accuracy_distribution: HashMap<String, (usize, usize)>, // (correct, total)
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
        // Running average update
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
            .map_or(0.0, |&(correct, total)| {
                if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                }
            })
    }
}

/// The main router that holds model variants, strategy, and history.
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
        let uniform_weight = if n == 0 { 0.0 } else { 1.0 / n as f64 };
        Self {
            variants,
            strategy,
            decisions: Vec::new(),
            metrics: RoutingMetrics::new(),
            round_robin_counter: 0,
            adaptive_weights: vec![uniform_weight; n],
        }
    }

    fn set_strategy(&mut self, strategy: RoutingStrategy) {
        self.strategy = strategy;
    }

    /// Route a request to a model variant, returning its index.
    fn route(&mut self, request: &InferenceRequest) -> usize {
        let idx = match self.strategy {
            RoutingStrategy::RoundRobin => self.route_round_robin(),
            RoutingStrategy::LowestLatency => self.route_lowest_latency(request),
            RoutingStrategy::CostAware => self.route_cost_aware(request),
            RoutingStrategy::AccuracyWeighted => self.route_accuracy_weighted(request),
            RoutingStrategy::Adaptive => self.route_adaptive(request),
        };

        let shadow = self.pick_shadow(idx, request.id);

        let decision = RoutingDecision {
            request_id: request.id,
            selected_model: self.variants[idx].name.clone(),
            strategy_used: self.strategy,
            shadow_model: shadow.map(|s| self.variants[s].name.clone()),
        };
        self.decisions.push(decision);

        idx
    }

    fn route_round_robin(&mut self) -> usize {
        let idx = self.round_robin_counter % self.variants.len();
        self.round_robin_counter += 1;
        idx
    }

    fn route_lowest_latency(&self, request: &InferenceRequest) -> usize {
        // Pick the model with lowest latency that meets the SLA
        self.variants
            .iter()
            .enumerate()
            .filter(|(_, v)| v.avg_latency_ms <= request.max_latency_ms)
            .min_by(|(_, a), (_, b)| {
                a.avg_latency_ms
                    .partial_cmp(&b.avg_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i) // fallback to first if none meets SLA
    }

    fn route_cost_aware(&self, request: &InferenceRequest) -> usize {
        // Pick the cheapest model within budget that meets latency SLA
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
                    // No model meets both constraints; pick cheapest overall
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
        // For high-priority: pick highest accuracy within SLA
        // For low-priority: pick cheapest within SLA
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
                .map_or(
                    // fallback: highest accuracy regardless of latency
                    self.variants
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.accuracy
                                .partial_cmp(&b.accuracy)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or(0, |(i, _)| i),
                    |(i, _)| i,
                ),
            Priority::Medium => {
                // Balanced: score = accuracy * 0.5 + (1 - normalized_cost) * 0.3 + (1 - normalized_latency) * 0.2
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
                        let score_a = Self::balanced_score(a, max_cost, max_lat);
                        let score_b = Self::balanced_score(b, max_cost, max_lat);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(i, _)| i)
            }
            Priority::Low => self.route_cost_aware(request),
        }
    }

    fn balanced_score(variant: &ModelVariant, max_cost: f64, max_latency: f64) -> f64 {
        let norm_cost = if max_cost > 0.0 {
            variant.cost_per_request / max_cost
        } else {
            0.0
        };
        let norm_lat = if max_latency > 0.0 {
            variant.avg_latency_ms / max_latency
        } else {
            0.0
        };
        variant.accuracy * 0.5 + (1.0 - norm_cost) * 0.3 + (1.0 - norm_lat) * 0.2
    }

    fn route_adaptive(&self, request: &InferenceRequest) -> usize {
        // Use adaptive weights derived from performance feedback.
        // Deterministic selection based on request id mapped to weight buckets.
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

    /// Pick a shadow model (if any) for comparison traffic.
    /// Shadow the most accurate model if it differs from primary.
    fn pick_shadow(&self, primary_idx: usize, request_id: u64) -> Option<usize> {
        // Shadow 50% of requests (deterministic)
        let mut h = DefaultHasher::new();
        ("shadow", request_id).hash(&mut h);
        let should_shadow = (h.finish() % 2) == 0;
        if !should_shadow {
            return None;
        }

        // Shadow the highest-accuracy model if different from primary
        let best_accuracy_idx = self
            .variants
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.accuracy
                    .partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);

        if best_accuracy_idx == primary_idx {
            None
        } else {
            Some(best_accuracy_idx)
        }
    }

    /// Record observed performance and update adaptive weights.
    fn record_observation(&mut self, obs: &PerformanceObservation) {
        self.metrics.record(obs);
    }

    /// Update adaptive weights based on collected metrics.
    /// Models with better accuracy and lower latency get higher weight.
    fn update_adaptive_weights(&mut self) {
        if self.variants.is_empty() {
            return;
        }

        let mut scores = Vec::with_capacity(self.variants.len());
        for variant in &self.variants {
            let observed_acc = self.metrics.model_accuracy(&variant.name);
            // Use observed accuracy if available, otherwise fall back to declared
            let acc = if observed_acc > 0.0 {
                observed_acc
            } else {
                variant.accuracy
            };
            // Score: accuracy / (latency * cost), higher is better
            let latency_factor = variant.avg_latency_ms.max(1.0);
            let cost_factor = variant.cost_per_request.max(0.001);
            let score = acc / (latency_factor * cost_factor);
            scores.push(score);
        }

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

    fn decisions_for_model(&self, model_name: &str) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.selected_model == model_name)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Deterministic simulation helpers
// ---------------------------------------------------------------------------

fn simulate_latency(variant: &ModelVariant, request_id: u64) -> f64 {
    let mut h = DefaultHasher::new();
    ("latency", variant.name.as_str(), request_id).hash(&mut h);
    let jitter = (h.finish() % 20) as f64 - 10.0; // +/- 10ms
    (variant.avg_latency_ms + jitter).max(1.0)
}

fn simulate_accuracy(variant: &ModelVariant, request_id: u64) -> bool {
    let mut h = DefaultHasher::new();
    ("accuracy", variant.name.as_str(), request_id).hash(&mut h);
    let threshold = h.finish() as f64 / u64::MAX as f64;
    threshold < variant.accuracy
}

fn make_default_variants() -> Vec<ModelVariant> {
    vec![
        ModelVariant::new(VARIANT_NAMES[0], 10.0, 0.82, 0.001), // fast-small
        ModelVariant::new(VARIANT_NAMES[1], 50.0, 0.91, 0.005), // balanced
        ModelVariant::new(VARIANT_NAMES[2], 200.0, 0.97, 0.020), // accurate-large
    ]
}

// ---------------------------------------------------------------------------
// Helper functions extracted from main() to reduce cyclomatic complexity
// ---------------------------------------------------------------------------

/// Section 1: Print the model variant registry table.
fn print_variant_registry(variants: &[ModelVariant]) {
    println!("1. Model Variant Registry");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>16} {:>12} {:>10} {:>10}",
        "Model", "Latency(ms)", "Accuracy", "Cost($)"
    );
    println!("   {}", "\u{2500}".repeat(52));
    for v in variants {
        println!(
            "   {:>16} {:>12.1} {:>9.0}% {:>10.3}",
            v.name,
            v.avg_latency_ms,
            v.accuracy * 100.0,
            v.cost_per_request
        );
    }
    println!("   Registered {} model variants\n", variants.len());
}

/// Section 2: Run round-robin routing baseline and print results.
fn run_round_robin_baseline(variants: &[ModelVariant]) {
    println!("2. Round-Robin Routing Baseline");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::RoundRobin);
    let requests: Vec<InferenceRequest> = (0..30)
        .map(|i| InferenceRequest::new(i, Priority::Medium, 500.0, 1.0))
        .collect();

    for req in &requests {
        let idx = router.route(req);
        let variant = &router.variants[idx];
        let latency = simulate_latency(variant, req.id);
        let accurate = simulate_accuracy(variant, req.id);
        router.record_observation(&PerformanceObservation {
            model_name: variant.name.clone(),
            latency_ms: latency,
            was_accurate: accurate,
            cost: variant.cost_per_request,
        });
    }

    println!("   Routed {} requests via RoundRobin", requests.len());
    for name in VARIANT_NAMES {
        let count = router.decisions_for_model(name);
        println!(
            "   {name}: {count} requests ({:.0}%)",
            count as f64 / requests.len() as f64 * 100.0
        );
    }
    println!(
        "   Avg latency: {:.1}ms | Avg cost: ${:.4}",
        router.metrics.avg_latency, router.metrics.avg_cost
    );
    println!();
}

/// Section 3: Run latency-optimized routing and print results.
fn run_latency_optimized_routing(variants: &[ModelVariant]) {
    println!("3. Latency-Optimized Routing");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::LowestLatency);

    let latency_requests = vec![
        InferenceRequest::new(100, Priority::High, 20.0, 1.0), // tight SLA -> fast-small
        InferenceRequest::new(101, Priority::Medium, 100.0, 1.0), // relaxed -> could be fast or balanced
        InferenceRequest::new(102, Priority::Low, 500.0, 1.0), // very relaxed -> fast-small (lowest latency)
    ];

    println!(
        "   {:>4} {:>10} {:>14} {:>16}",
        "ID", "MaxLat(ms)", "Selected", "ActualLat(ms)"
    );
    println!("   {}", "\u{2500}".repeat(48));
    for req in &latency_requests {
        let idx = router.route(req);
        let variant = &router.variants[idx];
        let latency = simulate_latency(variant, req.id);
        println!(
            "   {:>4} {:>10.0} {:>14} {:>16.1}",
            req.id, req.max_latency_ms, variant.name, latency
        );
        router.record_observation(&PerformanceObservation {
            model_name: variant.name.clone(),
            latency_ms: latency,
            was_accurate: simulate_accuracy(variant, req.id),
            cost: variant.cost_per_request,
        });
    }
    println!();
}

/// Section 4: Run cost-aware routing and print results.
fn run_cost_aware_routing(variants: &[ModelVariant]) {
    println!("4. Cost-Aware Routing with Budget Constraints");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::CostAware);

    let cost_requests = vec![
        InferenceRequest::new(200, Priority::Low, 500.0, 0.002), // tight budget -> fast-small
        InferenceRequest::new(201, Priority::Medium, 500.0, 0.010), // moderate budget
        InferenceRequest::new(202, Priority::High, 500.0, 0.050), // generous budget
        InferenceRequest::new(203, Priority::Low, 30.0, 0.002),  // tight budget + tight latency
    ];

    println!(
        "   {:>4} {:>10} {:>10} {:>16} {:>10}",
        "ID", "MaxCost($)", "MaxLat", "Selected", "Cost($)"
    );
    println!("   {}", "\u{2500}".repeat(58));
    for req in &cost_requests {
        let idx = router.route(req);
        let variant = &router.variants[idx];
        println!(
            "   {:>4} {:>10.3} {:>9.0}ms {:>16} {:>10.3}",
            req.id, req.max_cost, req.max_latency_ms, variant.name, variant.cost_per_request
        );
        router.record_observation(&PerformanceObservation {
            model_name: variant.name.clone(),
            latency_ms: simulate_latency(variant, req.id),
            was_accurate: simulate_accuracy(variant, req.id),
            cost: variant.cost_per_request,
        });
    }
    println!(
        "   Total cost: ${:.4}",
        router.metrics.avg_cost * router.metrics.total_requests as f64
    );
    println!();
}

/// Section 4b: Run accuracy-weighted routing by priority and print results.
fn run_accuracy_weighted_routing(variants: &[ModelVariant]) {
    println!("   Accuracy-Weighted Routing by Priority:");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::AccuracyWeighted);

    let priority_requests = vec![
        InferenceRequest::new(250, Priority::High, 500.0, 1.0),
        InferenceRequest::new(251, Priority::Medium, 500.0, 1.0),
        InferenceRequest::new(252, Priority::Low, 500.0, 1.0),
    ];

    println!(
        "   {:>4} {:>8} {:>16} {:>10}",
        "ID", "Priority", "Selected", "Accuracy"
    );
    println!("   {}", "\u{2500}".repeat(42));
    for req in &priority_requests {
        let idx = router.route(req);
        let variant = &router.variants[idx];
        println!(
            "   {:>4} {:>8} {:>16} {:>9.0}%",
            req.id,
            req.priority,
            variant.name,
            variant.accuracy * 100.0
        );
    }
    println!();
}

/// Section 5: Run shadow traffic comparison and print results.
fn run_shadow_traffic_comparison(variants: &[ModelVariant]) {
    println!("5. Shadow Traffic Comparison");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::LowestLatency);

    let shadow_requests: Vec<InferenceRequest> = (300..320)
        .map(|i| InferenceRequest::new(i, Priority::Medium, 100.0, 1.0))
        .collect();

    let mut shadow_comparisons = Vec::new();
    for req in &shadow_requests {
        let idx = router.route(req);
        let primary = &router.variants[idx];
        let primary_acc = simulate_accuracy(primary, req.id);

        // Check if there is a shadow decision
        let last_decision = router.decisions.last().unwrap();
        if let Some(ref shadow_name) = last_decision.shadow_model {
            let shadow_variant = router
                .variants
                .iter()
                .find(|v| v.name == *shadow_name)
                .unwrap();
            let shadow_acc = simulate_accuracy(shadow_variant, req.id);
            shadow_comparisons.push((
                req.id,
                primary.name.clone(),
                primary_acc,
                shadow_name.clone(),
                shadow_acc,
            ));
        }

        router.record_observation(&PerformanceObservation {
            model_name: primary.name.clone(),
            latency_ms: simulate_latency(primary, req.id),
            was_accurate: primary_acc,
            cost: primary.cost_per_request,
        });
    }

    println!(
        "   {} requests routed, {} with shadow traffic ({:.0}%)",
        shadow_requests.len(),
        router.shadow_count(),
        router.shadow_count() as f64 / shadow_requests.len() as f64 * 100.0
    );

    if !shadow_comparisons.is_empty() {
        print_shadow_comparison_table(&shadow_comparisons);
    }
    println!();
}

/// Print the shadow comparison table and win/loss tally.
fn print_shadow_comparison_table(shadow_comparisons: &[(u64, String, bool, String, bool)]) {
    println!(
        "\n   {:>4} {:>14} {:>8} {:>14} {:>8}",
        "ID", "Primary", "Correct", "Shadow", "Correct"
    );
    println!("   {}", "\u{2500}".repeat(52));
    for (id, primary, p_acc, shadow, s_acc) in shadow_comparisons.iter().take(8) {
        println!(
            "   {:>4} {:>14} {:>8} {:>14} {:>8}",
            id,
            primary,
            if *p_acc { "yes" } else { "no" },
            shadow,
            if *s_acc { "yes" } else { "no" }
        );
    }
    // Tally shadow wins
    let shadow_wins = shadow_comparisons
        .iter()
        .filter(|(_, _, p, _, s)| !p && *s)
        .count();
    let primary_wins = shadow_comparisons
        .iter()
        .filter(|(_, _, p, _, s)| *p && !s)
        .count();
    let both_correct = shadow_comparisons
        .iter()
        .filter(|(_, _, p, _, s)| *p && *s)
        .count();
    println!(
        "\n   Shadow wins: {} | Primary wins: {} | Both correct: {} | Total shadow: {}",
        shadow_wins,
        primary_wins,
        both_correct,
        shadow_comparisons.len()
    );
}

/// Section 6: Run adaptive routing with feedback loop and print results.
fn run_adaptive_routing(variants: &[ModelVariant]) {
    println!("6. Adaptive Routing with Feedback Loop");
    println!("   ─────────────────────────────────────────");

    let mut router = ModelRouter::new(variants.to_vec(), RoutingStrategy::RoundRobin);
    // Demonstrate strategy switching
    router.set_strategy(RoutingStrategy::Adaptive);

    // Phase 1: initial uniform weights
    println!("   Initial weights:");
    for (i, w) in router.adaptive_weights.iter().enumerate() {
        println!("     {}: {:.3}", router.variants[i].name, w);
    }

    // Run some requests with initial weights
    route_and_observe(&mut router, 400, 50);

    println!("\n   After 50 requests (before adaptation):");
    for name in VARIANT_NAMES {
        let count = router.decisions_for_model(name);
        let acc = router.metrics.model_accuracy(name);
        println!(
            "     {name}: {count} routed, observed accuracy {:.1}%",
            acc * 100.0
        );
    }

    // Phase 2: update weights from observations
    router.update_adaptive_weights();
    println!("\n   Updated adaptive weights (feedback-adjusted):");
    for (i, w) in router.adaptive_weights.iter().enumerate() {
        println!("     {}: {:.3}", router.variants[i].name, w);
    }

    // Run more requests with updated weights
    let pre_count = router.decision_count();
    route_and_observe(&mut router, 450, 50);

    println!("\n   After 50 more requests (post-adaptation):");
    for name in VARIANT_NAMES {
        let post_count = router
            .decisions
            .iter()
            .skip(pre_count)
            .filter(|d| d.selected_model == name)
            .count();
        println!("     {name}: {post_count} routed");
    }

    print_final_metrics(&router);
    print_audit_trail(&router);
    println!();
}

/// Route a batch of requests through the router and record observations.
fn route_and_observe(router: &mut ModelRouter, start_id: u64, count: u64) {
    for i in 0..count {
        let req = InferenceRequest::new(start_id + i, Priority::Medium, 500.0, 1.0);
        let idx = router.route(&req);
        let variant = &router.variants[idx];
        router.record_observation(&PerformanceObservation {
            model_name: variant.name.clone(),
            latency_ms: simulate_latency(variant, req.id),
            was_accurate: simulate_accuracy(variant, req.id),
            cost: variant.cost_per_request,
        });
    }
}

/// Print final aggregate metrics.
fn print_final_metrics(router: &ModelRouter) {
    println!("\n   Final aggregate metrics:");
    println!("     Total requests: {}", router.metrics.total_requests);
    println!("     Avg latency: {:.1}ms", router.metrics.avg_latency);
    println!("     Avg cost: ${:.4}", router.metrics.avg_cost);
    println!("     Audit trail entries: {}", router.decision_count());
    println!(
        "     Shadow traffic: {} ({:.0}%)",
        router.shadow_count(),
        router.shadow_count() as f64 / router.decision_count() as f64 * 100.0
    );
}

/// Print the last 5 entries from the audit trail.
fn print_audit_trail(router: &ModelRouter) {
    println!("\n   Audit trail (last 5 decisions):");
    println!(
        "     {:>4} {:>16} {:>16} {:>14}",
        "Req", "Model", "Strategy", "Shadow"
    );
    println!("     {}", "\u{2500}".repeat(54));
    for d in router.decisions.iter().rev().take(5) {
        println!(
            "     {:>4} {:>16} {:>16} {:>14}",
            d.request_id,
            d.selected_model,
            d.strategy_used,
            d.shadow_model.as_deref().unwrap_or("-")
        );
    }
}

fn main() {
    println!("=== Model Selection Router Example ===\n");

    let variants = make_default_variants();

    print_variant_registry(&variants);
    run_round_robin_baseline(&variants);
    run_latency_optimized_routing(&variants);
    run_cost_aware_routing(&variants);
    run_accuracy_weighted_routing(&variants);
    run_shadow_traffic_comparison(&variants);
    run_adaptive_routing(&variants);

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_variant_creation() {
        let v = ModelVariant::new("test-model", 25.0, 0.95, 0.01);
        assert_eq!(v.name, "test-model");
        assert!((v.avg_latency_ms - 25.0).abs() < f64::EPSILON);
        assert!((v.accuracy - 0.95).abs() < f64::EPSILON);
        assert!((v.cost_per_request - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_routing_strategy_display() {
        assert_eq!(RoutingStrategy::RoundRobin.to_string(), "RoundRobin");
        assert_eq!(RoutingStrategy::LowestLatency.to_string(), "LowestLatency");
        assert_eq!(RoutingStrategy::CostAware.to_string(), "CostAware");
        assert_eq!(
            RoutingStrategy::AccuracyWeighted.to_string(),
            "AccuracyWeighted"
        );
        assert_eq!(RoutingStrategy::Adaptive.to_string(), "Adaptive");
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(Priority::High.to_string(), "High");
        assert_eq!(Priority::Medium.to_string(), "Medium");
        assert_eq!(Priority::Low.to_string(), "Low");
    }

    #[test]
    fn test_round_robin_cycles_evenly() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::RoundRobin);

        let mut counts = [0_usize; NUM_VARIANTS];
        for i in 0..30_u64 {
            let req = InferenceRequest::new(i, Priority::Medium, 500.0, 1.0);
            let idx = router.route(&req);
            counts[idx] += 1;
        }
        // 30 requests / 3 models = 10 each
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 10, "Model {} got {} instead of 10", i, count);
        }
    }

    #[test]
    fn test_lowest_latency_picks_fastest() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::LowestLatency);
        let req = InferenceRequest::new(1, Priority::Medium, 500.0, 1.0);
        let idx = router.route(&req);
        assert_eq!(idx, 0, "Should pick fast-small (idx 0)");
    }

    #[test]
    fn test_lowest_latency_respects_sla() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::LowestLatency);
        // SLA of 5ms: no model meets it, should fallback to 0
        let req = InferenceRequest::new(1, Priority::High, 5.0, 1.0);
        let idx = router.route(&req);
        assert_eq!(idx, 0, "Fallback to first model when none meets SLA");
    }

    #[test]
    fn test_cost_aware_picks_cheapest_in_budget() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::CostAware);
        // Budget allows any model, should pick cheapest
        let req = InferenceRequest::new(1, Priority::Medium, 500.0, 1.0);
        let idx = router.route(&req);
        assert_eq!(router.variants[idx].name, "fast-small");
    }

    #[test]
    fn test_cost_aware_respects_budget() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::CostAware);
        // Very small budget
        let req = InferenceRequest::new(1, Priority::Medium, 500.0, 0.002);
        let idx = router.route(&req);
        assert_eq!(
            router.variants[idx].name, "fast-small",
            "Should pick fast-small within $0.002 budget"
        );
    }

    #[test]
    fn test_accuracy_weighted_high_priority_picks_best() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::AccuracyWeighted);
        let req = InferenceRequest::new(1, Priority::High, 500.0, 1.0);
        let idx = router.route(&req);
        assert_eq!(
            router.variants[idx].name, "accurate-large",
            "High priority should pick most accurate model"
        );
    }

    #[test]
    fn test_accuracy_weighted_low_priority_picks_cheapest() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::AccuracyWeighted);
        let req = InferenceRequest::new(1, Priority::Low, 500.0, 1.0);
        let idx = router.route(&req);
        assert_eq!(
            router.variants[idx].name, "fast-small",
            "Low priority should pick cheapest model"
        );
    }

    #[test]
    fn test_adaptive_initial_uniform_weights() {
        let variants = make_default_variants();
        let router = ModelRouter::new(variants, RoutingStrategy::Adaptive);
        let expected = 1.0 / NUM_VARIANTS as f64;
        for w in &router.adaptive_weights {
            assert!(
                (w - expected).abs() < 1e-10,
                "Initial weight should be {expected}, got {w}"
            );
        }
    }

    #[test]
    fn test_adaptive_weights_update_from_feedback() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::Adaptive);

        // Simulate observations: accurate-large always correct, fast-small never
        for i in 0..20_u64 {
            router.record_observation(&PerformanceObservation {
                model_name: "accurate-large".to_string(),
                latency_ms: 200.0,
                was_accurate: true,
                cost: 0.020,
            });
            router.record_observation(&PerformanceObservation {
                model_name: "fast-small".to_string(),
                latency_ms: 10.0,
                was_accurate: i < 5, // 25% accuracy
                cost: 0.001,
            });
        }
        router.update_adaptive_weights();

        // fast-small should have non-trivial weight due to low latency*cost even with lower accuracy
        let sum: f64 = router.adaptive_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {sum}"
        );
        // All weights should be positive
        for w in &router.adaptive_weights {
            assert!(*w > 0.0, "Weight should be positive");
        }
    }

    #[test]
    fn test_shadow_traffic_deterministic() {
        let variants = make_default_variants();
        let mut router1 = ModelRouter::new(variants.clone(), RoutingStrategy::RoundRobin);
        let mut router2 = ModelRouter::new(variants, RoutingStrategy::RoundRobin);

        for i in 0..20_u64 {
            let req = InferenceRequest::new(i, Priority::Medium, 500.0, 1.0);
            router1.route(&req);
            router2.route(&req);
        }

        assert_eq!(router1.shadow_count(), router2.shadow_count());
        for (d1, d2) in router1.decisions.iter().zip(router2.decisions.iter()) {
            assert_eq!(d1.shadow_model, d2.shadow_model);
        }
    }

    #[test]
    fn test_routing_metrics_running_average() {
        let mut metrics = RoutingMetrics::new();
        metrics.record(&PerformanceObservation {
            model_name: "m1".to_string(),
            latency_ms: 10.0,
            was_accurate: true,
            cost: 0.01,
        });
        metrics.record(&PerformanceObservation {
            model_name: "m1".to_string(),
            latency_ms: 20.0,
            was_accurate: false,
            cost: 0.03,
        });
        assert_eq!(metrics.total_requests, 2);
        assert!((metrics.avg_latency - 15.0).abs() < 1e-10);
        assert!((metrics.avg_cost - 0.02).abs() < 1e-10);
        assert!((metrics.model_accuracy("m1") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_routing_metrics_empty() {
        let metrics = RoutingMetrics::new();
        assert_eq!(metrics.total_requests, 0);
        assert!(metrics.avg_latency.abs() < 1e-10);
        assert!(metrics.model_accuracy("nonexistent").abs() < 1e-10);
    }

    #[test]
    fn test_audit_trail_records_all_decisions() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::RoundRobin);
        for i in 0..15_u64 {
            let req = InferenceRequest::new(i, Priority::Medium, 500.0, 1.0);
            router.route(&req);
        }
        assert_eq!(router.decision_count(), 15);
        for (i, d) in router.decisions.iter().enumerate() {
            assert_eq!(d.request_id, i as u64);
            assert_eq!(d.strategy_used, RoutingStrategy::RoundRobin);
        }
    }

    #[test]
    fn test_set_strategy_changes_behavior() {
        let variants = make_default_variants();
        let mut router = ModelRouter::new(variants, RoutingStrategy::RoundRobin);

        let req = InferenceRequest::new(0, Priority::Medium, 500.0, 1.0);
        router.route(&req);
        assert_eq!(
            router.decisions.last().unwrap().strategy_used,
            RoutingStrategy::RoundRobin
        );

        router.set_strategy(RoutingStrategy::LowestLatency);
        let req = InferenceRequest::new(1, Priority::Medium, 500.0, 1.0);
        router.route(&req);
        assert_eq!(
            router.decisions.last().unwrap().strategy_used,
            RoutingStrategy::LowestLatency
        );
    }

    #[test]
    fn test_simulate_latency_deterministic() {
        let v = ModelVariant::new("test", 50.0, 0.9, 0.01);
        let l1 = simulate_latency(&v, 42);
        let l2 = simulate_latency(&v, 42);
        assert!((l1 - l2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simulate_accuracy_deterministic() {
        let v = ModelVariant::new("test", 50.0, 0.9, 0.01);
        assert_eq!(simulate_accuracy(&v, 42), simulate_accuracy(&v, 42));
    }

    #[test]
    fn test_balanced_score_computation() {
        let v = ModelVariant::new("test", 50.0, 0.90, 0.010);
        let score = ModelRouter::balanced_score(&v, 0.020, 200.0);
        // accuracy * 0.5 + (1 - 0.010/0.020) * 0.3 + (1 - 50/200) * 0.2
        // = 0.45 + 0.5*0.3 + 0.75*0.2 = 0.45 + 0.15 + 0.15 = 0.75
        assert!((score - 0.75).abs() < 1e-10, "Expected 0.75, got {score}");
    }
}
