#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerlessModel {
    pub name: String,
    pub size_mb: u32,
    pub load_time_us: u64,
    pub inference_time_us: u64,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WarmupStrategy {
    Eager,
    Lazy,
    Tiered,
    PriorityBased,
}

impl WarmupStrategy {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Eager => "Eager",
            Self::Lazy => "Lazy",
            Self::Tiered => "Tiered",
            Self::PriorityBased => "PriorityBased",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerState {
    pub models_loaded: u32,
    pub memory_used_mb: u32,
    pub memory_limit_mb: u32,
    pub last_request_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupPlan {
    pub strategy: WarmupStrategy,
    pub load_order: Vec<String>,
    pub estimated_warmup_time_us: u64,
    pub memory_required_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMetrics {
    pub first_request_us: u64,
    pub warm_request_us: u64,
    pub overhead_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepAliveResult {
    pub idle_timeout_s: u32,
    pub ping_interval_s: u32,
    pub total_pings: u32,
    pub cold_starts_avoided: u32,
    pub cost_saving_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupReport {
    pub baseline_metrics: ColdStartMetrics,
    pub strategies: Vec<StrategyResult>,
    pub keep_alive: KeepAliveResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    pub strategy: String,
    pub warmup_time_us: u64,
    pub memory_required_mb: u32,
    pub models_loaded: u32,
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// Generate a deterministic set of sample models for demonstration.
pub fn sample_models() -> Vec<ServerlessModel> {
    vec![
        ServerlessModel {
            name: "fraud-detector".to_string(),
            size_mb: 120,
            load_time_us: 15_000,
            inference_time_us: 500,
            priority: 1,
        },
        ServerlessModel {
            name: "sentiment-analyzer".to_string(),
            size_mb: 80,
            load_time_us: 10_000,
            inference_time_us: 300,
            priority: 2,
        },
        ServerlessModel {
            name: "recommendation-engine".to_string(),
            size_mb: 200,
            load_time_us: 25_000,
            inference_time_us: 800,
            priority: 3,
        },
        ServerlessModel {
            name: "spam-classifier".to_string(),
            size_mb: 50,
            load_time_us: 6_000,
            inference_time_us: 200,
            priority: 2,
        },
        ServerlessModel {
            name: "image-tagger".to_string(),
            size_mb: 300,
            load_time_us: 40_000,
            inference_time_us: 1_200,
            priority: 4,
        },
    ]
}

/// Deterministic hash helper using `DefaultHasher`.
pub fn deterministic_hash(seed: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    hasher.finish()
}

// Measure baseline cold start by summing all model load times (cold) vs
/// inference-only (warm).
pub fn measure_cold_start_baseline(models: &[ServerlessModel]) -> ColdStartMetrics {
    let total_load: u64 = models.iter().map(|m| m.load_time_us).sum();
    let total_inference: u64 = models.iter().map(|m| m.inference_time_us).sum();

    let first_request_us = total_load + total_inference;
    let warm_request_us = total_inference;
    let overhead_ratio = if warm_request_us > 0 {
        first_request_us as f64 / warm_request_us as f64
    } else {
        1.0
    };

    ColdStartMetrics {
        first_request_us,
        warm_request_us,
        overhead_ratio,
    }
}

/// Build a warmup plan for the given strategy and memory limit.
pub fn build_warmup_plan(
    models: &[ServerlessModel],
    strategy: &WarmupStrategy,
    memory_limit_mb: u32,
) -> WarmupPlan {
    match *strategy {
        WarmupStrategy::Eager => build_eager_plan(models, memory_limit_mb),
        WarmupStrategy::Lazy => build_lazy_plan(),
        WarmupStrategy::Tiered => build_tiered_plan(models, memory_limit_mb),
        WarmupStrategy::PriorityBased => build_priority_plan(models, memory_limit_mb),
    }
}

/// Eager: load all models that fit into memory.
pub fn build_eager_plan(models: &[ServerlessModel], memory_limit_mb: u32) -> WarmupPlan {
    let mut load_order = Vec::new();
    let mut memory_used: u32 = 0;
    let mut warmup_time: u64 = 0;

    for model in models {
        if memory_used + model.size_mb <= memory_limit_mb {
            load_order.push(model.name.clone());
            memory_used += model.size_mb;
            warmup_time += model.load_time_us;
        }
    }

    WarmupPlan {
        strategy: WarmupStrategy::Eager,
        load_order,
        estimated_warmup_time_us: warmup_time,
        memory_required_mb: memory_used,
    }
}

/// Lazy: load nothing at warmup time; everything loads on first request.
pub fn build_lazy_plan() -> WarmupPlan {
    WarmupPlan {
        strategy: WarmupStrategy::Lazy,
        load_order: Vec::new(),
        estimated_warmup_time_us: 0,
        memory_required_mb: 0,
    }
}

/// Tiered: load small models first, defer large ones.
pub fn build_tiered_plan(models: &[ServerlessModel], memory_limit_mb: u32) -> WarmupPlan {
    let mut sorted: Vec<&ServerlessModel> = models.iter().collect();
    sorted.sort_by_key(|m| m.size_mb);

    let mut load_order = Vec::new();
    let mut memory_used: u32 = 0;
    let mut warmup_time: u64 = 0;

    for model in sorted {
        if memory_used + model.size_mb <= memory_limit_mb {
            load_order.push(model.name.clone());
            memory_used += model.size_mb;
            warmup_time += model.load_time_us;
        }
    }

    WarmupPlan {
        strategy: WarmupStrategy::Tiered,
        load_order,
        estimated_warmup_time_us: warmup_time,
        memory_required_mb: memory_used,
    }
}

/// Priority-based: load highest-priority (lowest number) models first.
pub fn build_priority_plan(models: &[ServerlessModel], memory_limit_mb: u32) -> WarmupPlan {
    let mut sorted: Vec<&ServerlessModel> = models.iter().collect();
    sorted.sort_by_key(|m| m.priority);

    let mut load_order = Vec::new();
    let mut memory_used: u32 = 0;
    let mut warmup_time: u64 = 0;

    for model in sorted {
        if memory_used + model.size_mb <= memory_limit_mb {
            load_order.push(model.name.clone());
            memory_used += model.size_mb;
            warmup_time += model.load_time_us;
        }
    }

    WarmupPlan {
        strategy: WarmupStrategy::PriorityBased,
        load_order,
        estimated_warmup_time_us: warmup_time,
        memory_required_mb: memory_used,
    }
}

// Plan warmup respecting the container memory limit. Returns the resulting
/// container state after loading as many models as possible by priority.
pub fn plan_memory_aware_warmup(
    models: &[ServerlessModel],
    memory_limit_mb: u32,
) -> ContainerState {
    let mut sorted: Vec<&ServerlessModel> = models.iter().collect();
    sorted.sort_by_key(|m| m.priority);

    let mut memory_used: u32 = 0;
    let mut models_loaded: u32 = 0;

    for model in sorted {
        if memory_used + model.size_mb <= memory_limit_mb {
            memory_used += model.size_mb;
            models_loaded += 1;
        }
    }

    ContainerState {
        models_loaded,
        memory_used_mb: memory_used,
        memory_limit_mb,
        last_request_time: 0,
    }
}

// Simulate a keep-alive ping strategy.
//
// `idle_timeout_s` is the time after which the container is recycled.
// `ping_interval_s` is how often we send keep-alive pings.
//
// We simulate one hour of traffic with deterministic request arrival based on
/// hashing.
pub fn simulate_keep_alive(
    models: &[ServerlessModel],
    idle_timeout_s: u32,
    ping_interval_s: u32,
) -> KeepAliveResult {
    let simulation_duration_s: u32 = 3600; // 1 hour
    let total_pings = if ping_interval_s > 0 {
        simulation_duration_s / ping_interval_s
    } else {
        0
    };

    // Deterministic request pattern: requests arrive at hashed intervals
    let mut cold_starts_without_ping: u32 = 0;
    let mut last_activity: u32 = 0;
    let num_requests = 10u32;

    for i in 0..num_requests {
        let hash = deterministic_hash(&format!("request-{}-{}", models.len(), i));
        let gap = (hash % (u64::from(idle_timeout_s) * 2)) as u32;
        let arrival = last_activity + gap;

        if arrival > last_activity + idle_timeout_s {
            cold_starts_without_ping += 1;
        }
        last_activity = arrival;
    }

    // With pings, the container never goes idle longer than ping_interval_s,
    // so cold starts are avoided as long as ping_interval_s < idle_timeout_s.
    let cold_starts_avoided = if ping_interval_s < idle_timeout_s {
        cold_starts_without_ping
    } else {
        0
    };

    let cost_saving_pct = if cold_starts_without_ping > 0 {
        (f64::from(cold_starts_avoided) / f64::from(cold_starts_without_ping)) * 100.0
    } else {
        0.0
    };

    KeepAliveResult {
        idle_timeout_s,
        ping_interval_s,
        total_pings,
        cold_starts_avoided,
        cost_saving_pct,
    }
}

pub fn save_report(path: &std::path::Path, report: &WarmupReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
