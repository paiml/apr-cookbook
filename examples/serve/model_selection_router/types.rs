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

pub const NUM_VARIANTS: usize = 3;
pub const VARIANT_NAMES: [&str; NUM_VARIANTS] = ["fast-small", "balanced", "accurate-large"];

#[derive(Clone, Debug)]
pub struct ModelVariant {
    pub name: String,
    pub avg_latency_ms: f64,
    pub accuracy: f64,
    pub cost_per_request: f64,
}

impl ModelVariant {
    pub fn new(name: &str, avg_latency_ms: f64, accuracy: f64, cost_per_request: f64) -> Self {
        Self {
            name: name.to_string(),
            avg_latency_ms,
            accuracy,
            cost_per_request,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingStrategy {
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
pub enum Priority {
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
pub struct InferenceRequest {
    pub id: u64,
    pub priority: Priority,
    pub max_latency_ms: f64,
    pub max_cost: f64,
}

impl InferenceRequest {
    pub fn new(id: u64, priority: Priority, max_latency_ms: f64, max_cost: f64) -> Self {
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
pub struct RoutingDecision {
    pub request_id: u64,
    pub selected_model: String,
    pub strategy_used: RoutingStrategy,
    pub shadow_model: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PerformanceObservation {
    pub model_name: String,
    pub latency_ms: f64,
    pub was_accurate: bool,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct RoutingMetrics {
    pub total_requests: usize,
    pub avg_latency: f64,
    pub avg_cost: f64,
    pub accuracy_distribution: HashMap<String, (usize, usize)>,
}

impl RoutingMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            avg_latency: 0.0,
            avg_cost: 0.0,
            accuracy_distribution: HashMap::new(),
        }
    }

    pub fn record(&mut self, obs: &PerformanceObservation) {
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

    pub fn model_accuracy(&self, model_name: &str) -> f64 {
        self.accuracy_distribution
            .get(model_name)
            .map_or(
                0.0,
                |&(c, t)| if t == 0 { 0.0 } else { c as f64 / t as f64 },
            )
    }
}

pub struct ModelRouter {
    pub variants: Vec<ModelVariant>,
    pub strategy: RoutingStrategy,
    pub decisions: Vec<RoutingDecision>,
    pub metrics: RoutingMetrics,
    pub round_robin_counter: usize,
    pub adaptive_weights: Vec<f64>,
}

impl ModelRouter {
    pub fn new(variants: Vec<ModelVariant>, strategy: RoutingStrategy) -> Self {
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

    pub fn route(&mut self, request: &InferenceRequest) -> usize {
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

    pub fn route_round_robin(&mut self) -> usize {
        let idx = self.round_robin_counter % self.variants.len();
        self.round_robin_counter += 1;
        idx
    }

    pub fn route_lowest_latency(&self, request: &InferenceRequest) -> usize {
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

    pub fn route_cost_aware(&self, request: &InferenceRequest) -> usize {
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

    pub fn route_accuracy_weighted(&self, request: &InferenceRequest) -> usize {
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

    pub fn balanced_score(v: &ModelVariant, max_cost: f64, max_lat: f64) -> f64 {
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

    pub fn route_adaptive(&self, request: &InferenceRequest) -> usize {
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

    pub fn pick_shadow(&self, primary_idx: usize, request_id: u64) -> Option<usize> {
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

    pub fn record_observation(&mut self, obs: &PerformanceObservation) {
        self.metrics.record(obs);
    }

    pub fn update_adaptive_weights(&mut self) {
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

    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }
    pub fn shadow_count(&self) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.shadow_model.is_some())
            .count()
    }
    pub fn decisions_for_model(&self, name: &str) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.selected_model == name)
            .count()
    }
}

pub fn simulate_latency(v: &ModelVariant, id: u64) -> f64 {
    let mut h = DefaultHasher::new();
    ("latency", v.name.as_str(), id).hash(&mut h);
    (v.avg_latency_ms + (h.finish() % 20) as f64 - 10.0).max(1.0)
}

pub fn simulate_accuracy(v: &ModelVariant, id: u64) -> bool {
    let mut h = DefaultHasher::new();
    ("accuracy", v.name.as_str(), id).hash(&mut h);
    (h.finish() as f64 / u64::MAX as f64) < v.accuracy
}

pub fn make_default_variants() -> Vec<ModelVariant> {
    vec![
        ModelVariant::new(VARIANT_NAMES[0], 10.0, 0.82, 0.001),
        ModelVariant::new(VARIANT_NAMES[1], 50.0, 0.91, 0.005),
        ModelVariant::new(VARIANT_NAMES[2], 200.0, 0.97, 0.020),
    ]
}

pub fn route_and_observe(router: &mut ModelRouter, start_id: u64, count: u64) {
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
