//! Inference Cost Tracking and Resource Monitoring for Production ML Systems
//!
//! Demonstrates per-request cost accounting, budget alerting, and cost trend
//! analysis for production inference workloads -- all with zero external
//! dependencies beyond `std`.
//!
//! # Techniques
//!
//! - **Per-request cost tracking**: Decompose each inference into compute,
//!   memory, and token costs using a configurable pricing model
//! - **Cost aggregation by model**: Attribute costs to individual models for
//!   capacity planning and chargeback
//! - **Per-client billing summary**: Roll up costs by client/tenant for
//!   multi-tenant billing and quota enforcement
//! - **Budget monitoring with alerts**: Threshold-based alerting at warn (80%)
//!   and critical (95%) levels with automatic notification
//! - **Cost trend analysis**: Linear extrapolation over rolling windows to
//!   forecast monthly spend and detect cost anomalies
//! - **ASCII cost visualization**: Terminal-friendly bar charts showing cost
//!   breakdown by model and client
//!
//! # Architecture
//!
//! ```text
//! +-------------------------------------------------------------------+
//! |               Inference Cost Tracking Pipeline                     |
//! +-------------------------------------------------------------------+
//! |                                                                   |
//! |  Requests -------> CostModel ----------> InferenceRecord          |
//! |    (model_a)           |                      |                   |
//! |    (model_b)           |                      v                   |
//! |    (model_c)           |               CostTracker                |
//! |                        v                      |                   |
//! |                   Per-Request             +---+---+               |
//! |                   Cost = compute          |       |               |
//! |                        + memory      By Model  By Client          |
//! |                        + tokens          |       |               |
//! |                                          v       v               |
//! |                                    BudgetMonitor                  |
//! |                                          |                        |
//! |                                          v                        |
//! |                                    CostForecast                   |
//! |                                    [trend + EOM projection]       |
//! +-------------------------------------------------------------------+
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example inference_cost_tracking
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Monitoring
//! - **Complexity**: Intermediate
//! - **Dependencies**: None (std only)
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Number of models in the simulation
const NUM_MODELS: usize = 3;

/// Model names
const MODEL_NAMES: [&str; NUM_MODELS] = ["gpt-small", "gpt-medium", "gpt-large"];

/// Number of clients in the simulation
const NUM_CLIENTS: usize = 4;

/// Client names
const CLIENT_NAMES: [&str; NUM_CLIENTS] = ["acme-corp", "globex-inc", "initech", "umbrella-co"];

/// Number of rolling windows for trend analysis
const NUM_WINDOWS: usize = 6;

/// Requests per rolling window
const REQUESTS_PER_WINDOW: usize = 100;

/// Width of ASCII bar chart (characters)
const BAR_WIDTH: usize = 40;

// ============================================================================
// Deterministic RNG
// ============================================================================

/// Deterministic pseudo-random number generator using `DefaultHasher`.
///
/// Produces repeatable sequences given the same seed, suitable for
/// simulation without pulling in external crate dependencies.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance state and return a pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.state = hasher.finish();
        self.state
    }

    /// Return a float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Return an integer in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

// ============================================================================
// Cost Model
// ============================================================================

/// Pricing configuration for inference cost computation.
///
/// Costs are modeled as the sum of three components:
/// - Compute: charged per millisecond of inference latency
/// - Memory: charged per megabyte of peak memory usage
/// - Tokens: charged per 1,000 input/output tokens
#[derive(Debug, Clone, Copy)]
struct CostModel {
    /// Cost per millisecond of compute time (USD)
    cost_per_ms: f64,
    /// Cost per megabyte of memory usage (USD)
    cost_per_mb: f64,
    /// Cost per 1,000 tokens processed (USD)
    cost_per_1k_tokens: f64,
}

impl CostModel {
    /// Compute the total cost for a single inference request.
    fn compute_cost(&self, latency_ms: f64, memory_mb: f64, tokens: u64) -> f64 {
        let compute_cost = self.cost_per_ms * latency_ms;
        let memory_cost = self.cost_per_mb * memory_mb;
        let token_cost = self.cost_per_1k_tokens * (tokens as f64 / 1000.0);
        compute_cost + memory_cost + token_cost
    }

    /// Compute the individual cost components for a single inference request.
    fn cost_breakdown(&self, latency_ms: f64, memory_mb: f64, tokens: u64) -> CostBreakdown {
        CostBreakdown {
            compute: self.cost_per_ms * latency_ms,
            memory: self.cost_per_mb * memory_mb,
            tokens: self.cost_per_1k_tokens * (tokens as f64 / 1000.0),
        }
    }
}

impl fmt::Display for CostModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compute=${:.6}/ms, memory=${:.6}/MB, tokens=${:.4}/1k",
            self.cost_per_ms, self.cost_per_mb, self.cost_per_1k_tokens
        )
    }
}

// ============================================================================
// Cost Breakdown
// ============================================================================

/// Individual cost components for a single inference request.
#[derive(Debug, Clone, Copy)]
struct CostBreakdown {
    /// Compute cost (USD)
    compute: f64,
    /// Memory cost (USD)
    memory: f64,
    /// Token cost (USD)
    tokens: f64,
}

impl CostBreakdown {
    /// Total cost across all components.
    fn total(&self) -> f64 {
        self.compute + self.memory + self.tokens
    }
}

impl fmt::Display for CostBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compute=${:.6}, memory=${:.6}, tokens=${:.6}, total=${:.6}",
            self.compute,
            self.memory,
            self.tokens,
            self.total()
        )
    }
}

// ============================================================================
// Inference Record
// ============================================================================

/// A single inference request with resource usage measurements.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct InferenceRecord {
    /// Model identifier
    model: String,
    /// Client/tenant identifier
    client: String,
    /// Inference latency in milliseconds
    latency_ms: f64,
    /// Peak memory usage in megabytes
    memory_mb: f64,
    /// Number of tokens processed (input + output)
    tokens: u64,
    /// Simulated timestamp (seconds since epoch)
    timestamp: u64,
    /// Computed total cost (USD)
    total_cost: f64,
}

impl fmt::Display for InferenceRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model={}, client={}, latency={:.1}ms, mem={:.1}MB, tokens={}, cost=${:.6}",
            self.model, self.client, self.latency_ms, self.memory_mb, self.tokens, self.total_cost
        )
    }
}

// ============================================================================
// Cost Tracker
// ============================================================================

/// Accumulates inference records and computes per-model and per-client costs.
struct CostTracker {
    records: Vec<InferenceRecord>,
}

impl CostTracker {
    fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Add an inference record.
    fn add(&mut self, record: InferenceRecord) {
        self.records.push(record);
    }

    /// Total number of recorded requests.
    fn count(&self) -> usize {
        self.records.len()
    }

    /// Total cost across all records.
    fn total_cost(&self) -> f64 {
        self.records.iter().map(|r| r.total_cost).sum()
    }

    /// Mean cost per request.
    fn mean_cost(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.total_cost() / self.records.len() as f64
    }

    /// Aggregate cost by model name.
    ///
    /// Returns a vector of `(model_name, request_count, total_cost, mean_cost)`.
    fn cost_by_model(&self) -> Vec<(String, usize, f64, f64)> {
        let mut model_costs: Vec<(String, usize, f64)> = Vec::new();

        for record in &self.records {
            if let Some(entry) = model_costs
                .iter_mut()
                .find(|(name, _, _)| *name == record.model)
            {
                entry.1 += 1;
                entry.2 += record.total_cost;
            } else {
                model_costs.push((record.model.clone(), 1, record.total_cost));
            }
        }

        model_costs
            .into_iter()
            .map(|(name, count, total)| {
                let mean = if count > 0 { total / count as f64 } else { 0.0 };
                (name, count, total, mean)
            })
            .collect()
    }

    /// Aggregate cost by client name.
    ///
    /// Returns a vector of `(client_name, request_count, total_cost, mean_cost)`.
    fn cost_by_client(&self) -> Vec<(String, usize, f64, f64)> {
        let mut client_costs: Vec<(String, usize, f64)> = Vec::new();

        for record in &self.records {
            if let Some(entry) = client_costs
                .iter_mut()
                .find(|(name, _, _)| *name == record.client)
            {
                entry.1 += 1;
                entry.2 += record.total_cost;
            } else {
                client_costs.push((record.client.clone(), 1, record.total_cost));
            }
        }

        client_costs
            .into_iter()
            .map(|(name, count, total)| {
                let mean = if count > 0 { total / count as f64 } else { 0.0 };
                (name, count, total, mean)
            })
            .collect()
    }

    /// Cost-per-prediction metric (same as mean cost, but named for ROI context).
    fn cost_per_prediction(&self) -> f64 {
        self.mean_cost()
    }

    /// Render an ASCII bar chart of cost breakdown by a given dimension.
    ///
    /// `entries` is a slice of `(label, total_cost)` pairs.
    fn ascii_cost_chart(entries: &[(String, f64)]) -> String {
        let max_cost = entries
            .iter()
            .map(|(_, c)| *c)
            .fold(0.0_f64, f64::max)
            .max(f64::EPSILON);

        let mut out = String::new();
        for (label, cost) in entries {
            let bar_len = ((*cost / max_cost) * BAR_WIDTH as f64) as usize;
            let bar: String = "#".repeat(bar_len);
            out.push_str(&format!(
                "   {:>14} |{:<width$}| ${:.4}\n",
                label,
                bar,
                cost,
                width = BAR_WIDTH
            ));
        }
        out
    }
}

// ============================================================================
// Budget Monitor
// ============================================================================

/// Alert level for budget monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AlertLevel {
    /// Within budget, no action needed
    Ok,
    /// Approaching budget (>= 80% consumed)
    Warn,
    /// Near budget exhaustion (>= 95% consumed)
    Critical,
    /// Budget exceeded (>= 100% consumed)
    Exceeded,
}

impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::Warn => write!(f, "WARN"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Exceeded => write!(f, "EXCEEDED"),
        }
    }
}

/// Monitors spending against a budget and raises alerts at thresholds.
struct BudgetMonitor {
    /// Total budget for the monitoring period (USD)
    budget: f64,
    /// Warning threshold as a fraction (e.g. 0.80 for 80%)
    warn_threshold: f64,
    /// Critical threshold as a fraction (e.g. 0.95 for 95%)
    critical_threshold: f64,
    /// Accumulated spend (USD)
    spent: f64,
}

impl BudgetMonitor {
    fn new(budget: f64, warn_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            budget,
            warn_threshold,
            critical_threshold,
            spent: 0.0,
        }
    }

    /// Record additional spending.
    fn record_spend(&mut self, amount: f64) {
        self.spent += amount;
    }

    /// Current spend (used in tests).
    #[allow(dead_code)]
    fn spent(&self) -> f64 {
        self.spent
    }

    /// Budget utilization as a fraction (0.0 to 1.0+).
    fn utilization(&self) -> f64 {
        if self.budget <= 0.0 {
            return if self.spent > 0.0 { f64::INFINITY } else { 0.0 };
        }
        self.spent / self.budget
    }

    /// Remaining budget (may be negative if exceeded).
    fn remaining(&self) -> f64 {
        self.budget - self.spent
    }

    /// Current alert level based on utilization.
    fn alert_level(&self) -> AlertLevel {
        let util = self.utilization();
        if util >= 1.0 {
            AlertLevel::Exceeded
        } else if util >= self.critical_threshold {
            AlertLevel::Critical
        } else if util >= self.warn_threshold {
            AlertLevel::Warn
        } else {
            AlertLevel::Ok
        }
    }
}

impl fmt::Display for BudgetMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "budget=${:.4}, spent=${:.4}, remaining=${:.4}, utilization={:.1}%, status={}",
            self.budget,
            self.spent,
            self.remaining(),
            self.utilization() * 100.0,
            self.alert_level()
        )
    }
}

// ============================================================================
// Cost Forecast
// ============================================================================

/// Linear cost trend extrapolation over rolling windows.
///
/// Fits a simple linear regression `cost = slope * window + intercept`
/// to the per-window cost totals and projects end-of-period spend.
#[derive(Debug, Clone)]
struct CostForecast {
    /// Per-window cost totals
    window_costs: Vec<f64>,
    /// Slope of the linear fit (cost change per window)
    slope: f64,
    /// Intercept of the linear fit
    intercept: f64,
}

impl CostForecast {
    /// Create a new forecast from per-window cost totals.
    fn from_window_costs(costs: &[f64]) -> Self {
        let (slope, intercept) = if costs.len() < 2 {
            (0.0, costs.first().copied().unwrap_or(0.0))
        } else {
            linear_regression(costs)
        };
        Self {
            window_costs: costs.to_vec(),
            slope,
            intercept,
        }
    }

    /// Predicted cost for a given window index.
    fn predict(&self, window_idx: usize) -> f64 {
        self.intercept + self.slope * window_idx as f64
    }

    /// Project total cost over `num_windows` windows.
    fn project_total(&self, num_windows: usize) -> f64 {
        (0..num_windows).map(|i| self.predict(i).max(0.0)).sum()
    }

    /// Trend direction label.
    fn trend_label(&self) -> &str {
        if self.slope > 1e-9 {
            "INCREASING"
        } else if self.slope < -1e-9 {
            "DECREASING"
        } else {
            "STABLE"
        }
    }

    /// Percentage change from first to last window.
    fn pct_change(&self) -> f64 {
        if self.window_costs.len() < 2 {
            return 0.0;
        }
        let first = self.window_costs[0];
        let last = self.window_costs[self.window_costs.len() - 1];
        if first.abs() < f64::EPSILON {
            return 0.0;
        }
        (last - first) / first * 100.0
    }
}

impl fmt::Display for CostForecast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "slope={:.6}/window, trend={}, pct_change={:+.1}%",
            self.slope,
            self.trend_label(),
            self.pct_change()
        )
    }
}

// ============================================================================
// Linear Regression Helper
// ============================================================================

/// Simple linear regression on equally-spaced data.
///
/// Given values `y[0], y[1], ..., y[n-1]` at indices `x = 0, 1, ..., n-1`,
/// returns `(slope, intercept)` minimizing sum of squared residuals.
fn linear_regression(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n < 2.0 {
        return (0.0, values.first().copied().unwrap_or(0.0));
    }

    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..values.len()).map(|i| (i as f64) * (i as f64)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return (0.0, sum_y / n);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

// ============================================================================
// Simulation Helpers
// ============================================================================

/// Simulate resource usage for a given model variant.
///
/// Each model has a different resource profile:
/// - `gpt-small`:  fast, low memory, few tokens
/// - `gpt-medium`: moderate across all dimensions
/// - `gpt-large`:  slow, high memory, many tokens
fn simulate_request(
    model_idx: usize,
    client_idx: usize,
    timestamp: u64,
    cost_model: &CostModel,
    rng: &mut DeterministicRng,
) -> InferenceRecord {
    let (base_latency, base_memory, base_tokens) = match model_idx {
        0 => (15.0, 256.0, 200.0),    // gpt-small
        1 => (45.0, 1024.0, 500.0),   // gpt-medium
        _ => (120.0, 4096.0, 1500.0), // gpt-large
    };

    // Add variance (roughly +/- 30%)
    let latency_ms = base_latency * (0.7 + rng.next_f64() * 0.6);
    let memory_mb = base_memory * (0.8 + rng.next_f64() * 0.4);
    let tokens = (base_tokens * (0.7 + rng.next_f64() * 0.6)) as u64;

    let total_cost = cost_model.compute_cost(latency_ms, memory_mb, tokens);

    InferenceRecord {
        model: MODEL_NAMES[model_idx].to_string(),
        client: CLIENT_NAMES[client_idx].to_string(),
        latency_ms,
        memory_mb,
        tokens,
        timestamp,
        total_cost,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Inference Cost Tracking & Resource Monitoring Example ===\n");

    let mut rng = DeterministicRng::new(42);

    // =========================================================================
    // 1. Define Cost Model
    // =========================================================================
    println!("1. Define Cost Model");
    println!("   -----------------------------------------------");

    let cost_model = CostModel {
        cost_per_ms: 0.000_002,      // $0.002 per second of compute
        cost_per_mb: 0.000_000_5,    // $0.0005 per GB of memory
        cost_per_1k_tokens: 0.000_3, // $0.30 per million tokens
    };

    println!("   Pricing configuration:");
    println!("     {cost_model}");
    println!();

    // Show example cost breakdown for each model
    println!("   Example cost breakdown (single request per model):");
    for (i, name) in MODEL_NAMES.iter().enumerate() {
        let record = simulate_request(i, 0, 0, &cost_model, &mut rng);
        let breakdown =
            cost_model.cost_breakdown(record.latency_ms, record.memory_mb, record.tokens);
        println!("     {name}: {breakdown}");
    }
    println!();

    // =========================================================================
    // 2. Track Individual Inference Requests
    // =========================================================================
    println!("2. Track Individual Inference Requests");
    println!("   -----------------------------------------------");

    let mut tracker = CostTracker::new();
    let total_requests = 600;

    for i in 0..total_requests {
        let model_idx = rng.next_usize(NUM_MODELS);
        let client_idx = rng.next_usize(NUM_CLIENTS);
        let timestamp = 1_700_000_000 + (i as u64) * 60; // 1 request per minute
        let record = simulate_request(model_idx, client_idx, timestamp, &cost_model, &mut rng);
        tracker.add(record);
    }

    println!("   Total requests tracked: {}", tracker.count());
    println!("   Total cost:             ${:.6}", tracker.total_cost());
    println!("   Mean cost per request:  ${:.6}", tracker.mean_cost());
    println!(
        "   Cost per prediction:    ${:.6}",
        tracker.cost_per_prediction()
    );
    println!();

    // Show first 5 records as examples
    println!("   Sample records (first 5):");
    for record in tracker.records.iter().take(5) {
        println!("     {record}");
    }
    println!();

    // =========================================================================
    // 3. Per-Model Cost Aggregation
    // =========================================================================
    println!("3. Per-Model Cost Aggregation");
    println!("   -----------------------------------------------");

    let model_costs = tracker.cost_by_model();
    println!(
        "   {:>14}  {:>8}  {:>12}  {:>12}",
        "Model", "Requests", "Total Cost", "Mean Cost"
    );
    println!("   {}", "-".repeat(54));
    for (name, count, total, mean) in &model_costs {
        println!(
            "   {:>14}  {:>8}  ${:>11.6}  ${:>11.6}",
            name, count, total, mean
        );
    }
    println!();

    // ASCII chart of model costs
    println!("   Cost by model:");
    let chart_entries: Vec<(String, f64)> = model_costs
        .iter()
        .map(|(name, _, total, _)| (name.clone(), *total))
        .collect();
    print!("{}", CostTracker::ascii_cost_chart(&chart_entries));
    println!();

    // =========================================================================
    // 4. Per-Client Billing Summary
    // =========================================================================
    println!("4. Per-Client Billing Summary");
    println!("   -----------------------------------------------");

    let client_costs = tracker.cost_by_client();
    println!(
        "   {:>14}  {:>8}  {:>12}  {:>12}",
        "Client", "Requests", "Total Cost", "Mean Cost"
    );
    println!("   {}", "-".repeat(54));
    for (name, count, total, mean) in &client_costs {
        println!(
            "   {:>14}  {:>8}  ${:>11.6}  ${:>11.6}",
            name, count, total, mean
        );
    }
    println!();

    // ASCII chart of client costs
    println!("   Cost by client:");
    let chart_entries: Vec<(String, f64)> = client_costs
        .iter()
        .map(|(name, _, total, _)| (name.clone(), *total))
        .collect();
    print!("{}", CostTracker::ascii_cost_chart(&chart_entries));
    println!();

    // =========================================================================
    // 5. Budget Monitoring with Alert Thresholds
    // =========================================================================
    println!("5. Budget Monitoring with Alert Thresholds");
    println!("   -----------------------------------------------");

    // Set a budget that the simulation will exceed
    let monthly_budget = tracker.total_cost() * 0.90;
    let mut budget_monitor = BudgetMonitor::new(monthly_budget, 0.80, 0.95);

    println!("   Monthly budget: ${:.4}", monthly_budget);
    println!(
        "   Warn threshold: {:.0}%",
        budget_monitor.warn_threshold * 100.0
    );
    println!(
        "   Critical threshold: {:.0}%",
        budget_monitor.critical_threshold * 100.0
    );
    println!();

    // Replay records and check budget at each quartile
    let quartile_size = tracker.count() / 4;
    for (i, record) in tracker.records.iter().enumerate() {
        budget_monitor.record_spend(record.total_cost);

        if (i + 1) % quartile_size == 0 || i == tracker.count() - 1 {
            let pct = (i + 1) as f64 / tracker.count() as f64 * 100.0;
            println!("   At {:.0}% of requests: {}", pct, budget_monitor);
        }
    }
    println!();

    let final_level = budget_monitor.alert_level();
    match final_level {
        AlertLevel::Exceeded => {
            println!(
                "   ALERT: Budget exceeded by ${:.4}!",
                -budget_monitor.remaining()
            );
            println!("   Action: Review high-cost models and consider rate limiting");
        }
        AlertLevel::Critical => {
            println!(
                "   ALERT: Budget nearly exhausted ({:.1}% used)",
                budget_monitor.utilization() * 100.0
            );
            println!("   Action: Throttle non-critical workloads immediately");
        }
        AlertLevel::Warn => {
            println!(
                "   WARNING: Budget consumption elevated ({:.1}% used)",
                budget_monitor.utilization() * 100.0
            );
            println!("   Action: Monitor closely and prepare contingency");
        }
        AlertLevel::Ok => {
            println!(
                "   Budget status: healthy ({:.1}% used)",
                budget_monitor.utilization() * 100.0
            );
        }
    }
    println!();

    // =========================================================================
    // 6. Cost Trend Analysis and Forecasting
    // =========================================================================
    println!("6. Cost Trend Analysis and Forecasting");
    println!("   -----------------------------------------------");

    // Simulate windows with increasing cost (simulating growing traffic)
    let mut window_costs: Vec<f64> = Vec::with_capacity(NUM_WINDOWS);
    let mut rng_trend = DeterministicRng::new(99);

    for window_id in 0..NUM_WINDOWS {
        let mut window_total = 0.0;
        let traffic_multiplier = 1.0 + (window_id as f64) * 0.15; // 15% growth per window

        for _ in 0..REQUESTS_PER_WINDOW {
            let model_idx = rng_trend.next_usize(NUM_MODELS);
            let client_idx = rng_trend.next_usize(NUM_CLIENTS);
            let record = simulate_request(model_idx, client_idx, 0, &cost_model, &mut rng_trend);
            window_total += record.total_cost * traffic_multiplier;
        }

        window_costs.push(window_total);
        println!(
            "   Window {}: ${:.6} (traffic multiplier: {:.2}x)",
            window_id, window_total, traffic_multiplier
        );
    }
    println!();

    let forecast = CostForecast::from_window_costs(&window_costs);
    println!("   Trend analysis: {forecast}");
    println!(
        "   Projected cost for next window: ${:.6}",
        forecast.predict(NUM_WINDOWS)
    );

    // Project total for a 30-window month
    let monthly_projection = forecast.project_total(30);
    println!(
        "   Projected 30-window monthly cost: ${:.4}",
        monthly_projection
    );

    if forecast.slope > 0.0 {
        println!("   Recommendation: Costs are increasing; review scaling policies");
    } else {
        println!("   Recommendation: Costs are stable or decreasing; no action needed");
    }

    println!("\n=== Example Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_model_compute_cost() {
        let model = CostModel {
            cost_per_ms: 0.001,
            cost_per_mb: 0.0001,
            cost_per_1k_tokens: 0.01,
        };
        // 100ms, 512MB, 2000 tokens
        let cost = model.compute_cost(100.0, 512.0, 2000);
        let expected = 0.001 * 100.0 + 0.0001 * 512.0 + 0.01 * 2.0;
        assert!(
            (cost - expected).abs() < 1e-10,
            "Expected {expected}, got {cost}"
        );
    }

    #[test]
    fn test_cost_model_zero_inputs() {
        let model = CostModel {
            cost_per_ms: 0.001,
            cost_per_mb: 0.0001,
            cost_per_1k_tokens: 0.01,
        };
        let cost = model.compute_cost(0.0, 0.0, 0);
        assert!(cost.abs() < 1e-10, "Zero inputs should produce zero cost");
    }

    #[test]
    fn test_cost_breakdown_components() {
        let model = CostModel {
            cost_per_ms: 0.002,
            cost_per_mb: 0.001,
            cost_per_1k_tokens: 0.05,
        };
        let bd = model.cost_breakdown(50.0, 256.0, 1000);
        assert!((bd.compute - 0.1).abs() < 1e-10);
        assert!((bd.memory - 0.256).abs() < 1e-10);
        assert!((bd.tokens - 0.05).abs() < 1e-10);
        assert!((bd.total() - 0.406).abs() < 1e-10);
    }

    #[test]
    fn test_cost_model_display() {
        let model = CostModel {
            cost_per_ms: 0.000_002,
            cost_per_mb: 0.000_000_5,
            cost_per_1k_tokens: 0.000_3,
        };
        let display = format!("{model}");
        assert!(
            display.contains("compute="),
            "Should contain compute: {display}"
        );
        assert!(
            display.contains("memory="),
            "Should contain memory: {display}"
        );
        assert!(
            display.contains("tokens="),
            "Should contain tokens: {display}"
        );
    }

    #[test]
    fn test_cost_breakdown_display() {
        let bd = CostBreakdown {
            compute: 0.1,
            memory: 0.2,
            tokens: 0.05,
        };
        let display = format!("{bd}");
        assert!(
            display.contains("total="),
            "Should contain total: {display}"
        );
    }

    #[test]
    fn test_cost_tracker_empty() {
        let tracker = CostTracker::new();
        assert_eq!(tracker.count(), 0);
        assert!(tracker.total_cost().abs() < f64::EPSILON);
        assert!(tracker.mean_cost().abs() < f64::EPSILON);
        assert!(tracker.cost_by_model().is_empty());
        assert!(tracker.cost_by_client().is_empty());
    }

    #[test]
    fn test_cost_tracker_aggregation() {
        let mut tracker = CostTracker::new();
        tracker.add(InferenceRecord {
            model: "model_a".to_string(),
            client: "client_x".to_string(),
            latency_ms: 10.0,
            memory_mb: 100.0,
            tokens: 500,
            timestamp: 0,
            total_cost: 1.0,
        });
        tracker.add(InferenceRecord {
            model: "model_a".to_string(),
            client: "client_y".to_string(),
            latency_ms: 20.0,
            memory_mb: 200.0,
            tokens: 1000,
            timestamp: 1,
            total_cost: 2.0,
        });
        tracker.add(InferenceRecord {
            model: "model_b".to_string(),
            client: "client_x".to_string(),
            latency_ms: 30.0,
            memory_mb: 300.0,
            tokens: 1500,
            timestamp: 2,
            total_cost: 3.0,
        });

        assert_eq!(tracker.count(), 3);
        assert!((tracker.total_cost() - 6.0).abs() < 1e-10);
        assert!((tracker.mean_cost() - 2.0).abs() < 1e-10);

        let by_model = tracker.cost_by_model();
        assert_eq!(by_model.len(), 2);

        let by_client = tracker.cost_by_client();
        assert_eq!(by_client.len(), 2);
    }

    #[test]
    fn test_cost_tracker_by_model_values() {
        let mut tracker = CostTracker::new();
        for i in 0..5 {
            tracker.add(InferenceRecord {
                model: "m1".to_string(),
                client: "c1".to_string(),
                latency_ms: 10.0,
                memory_mb: 100.0,
                tokens: 500,
                timestamp: i,
                total_cost: 1.0,
            });
        }
        let by_model = tracker.cost_by_model();
        assert_eq!(by_model.len(), 1);
        assert_eq!(by_model[0].1, 5); // count
        assert!((by_model[0].2 - 5.0).abs() < 1e-10); // total
        assert!((by_model[0].3 - 1.0).abs() < 1e-10); // mean
    }

    #[test]
    fn test_budget_monitor_ok() {
        let monitor = BudgetMonitor::new(100.0, 0.80, 0.95);
        assert_eq!(monitor.alert_level(), AlertLevel::Ok);
        assert!((monitor.utilization()).abs() < f64::EPSILON);
        assert!((monitor.remaining() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_monitor_thresholds() {
        let mut monitor = BudgetMonitor::new(100.0, 0.80, 0.95);

        monitor.record_spend(79.0);
        assert_eq!(monitor.alert_level(), AlertLevel::Ok);

        monitor.record_spend(2.0); // 81%
        assert_eq!(monitor.alert_level(), AlertLevel::Warn);

        monitor.record_spend(15.0); // 96%
        assert_eq!(monitor.alert_level(), AlertLevel::Critical);

        monitor.record_spend(5.0); // 101%
        assert_eq!(monitor.alert_level(), AlertLevel::Exceeded);
    }

    #[test]
    fn test_budget_monitor_display() {
        let monitor = BudgetMonitor::new(100.0, 0.80, 0.95);
        let display = format!("{monitor}");
        assert!(
            display.contains("budget="),
            "Should contain budget: {display}"
        );
        assert!(display.contains("OK"), "Should contain OK: {display}");
    }

    #[test]
    fn test_budget_monitor_zero_budget() {
        let mut monitor = BudgetMonitor::new(0.0, 0.80, 0.95);
        assert!((monitor.utilization()).abs() < f64::EPSILON);
        monitor.record_spend(1.0);
        assert!(monitor.utilization().is_infinite());
    }

    #[test]
    fn test_alert_level_ordering() {
        assert!(AlertLevel::Ok < AlertLevel::Warn);
        assert!(AlertLevel::Warn < AlertLevel::Critical);
        assert!(AlertLevel::Critical < AlertLevel::Exceeded);
    }

    #[test]
    fn test_linear_regression_constant() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let (slope, intercept) = linear_regression(&values);
        assert!(slope.abs() < 1e-10, "Constant data should have slope ~0");
        assert!(
            (intercept - 5.0).abs() < 1e-10,
            "Intercept should be ~5.0, got {intercept}"
        );
    }

    #[test]
    fn test_linear_regression_perfect_line() {
        // y = 2x + 1 at x = 0,1,2,3
        let values = vec![1.0, 3.0, 5.0, 7.0];
        let (slope, intercept) = linear_regression(&values);
        assert!(
            (slope - 2.0).abs() < 1e-10,
            "Slope should be 2.0, got {slope}"
        );
        assert!(
            (intercept - 1.0).abs() < 1e-10,
            "Intercept should be 1.0, got {intercept}"
        );
    }

    #[test]
    fn test_linear_regression_single_value() {
        let values = vec![42.0];
        let (slope, intercept) = linear_regression(&values);
        assert!(slope.abs() < 1e-10);
        assert!((intercept - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_forecast_stable() {
        let costs = vec![10.0, 10.0, 10.0, 10.0];
        let forecast = CostForecast::from_window_costs(&costs);
        assert_eq!(forecast.trend_label(), "STABLE");
        assert!(
            forecast.pct_change().abs() < 1e-10,
            "Stable series should have ~0% change"
        );
    }

    #[test]
    fn test_cost_forecast_increasing() {
        let costs = vec![10.0, 20.0, 30.0, 40.0];
        let forecast = CostForecast::from_window_costs(&costs);
        assert_eq!(forecast.trend_label(), "INCREASING");
        assert!(forecast.slope > 0.0);
        assert!(forecast.pct_change() > 0.0);
        // predict(4) should be around 50
        assert!((forecast.predict(4) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_forecast_decreasing() {
        let costs = vec![40.0, 30.0, 20.0, 10.0];
        let forecast = CostForecast::from_window_costs(&costs);
        assert_eq!(forecast.trend_label(), "DECREASING");
        assert!(forecast.slope < 0.0);
        assert!(forecast.pct_change() < 0.0);
    }

    #[test]
    fn test_cost_forecast_project_total() {
        let costs = vec![10.0, 10.0, 10.0];
        let forecast = CostForecast::from_window_costs(&costs);
        // Constant at 10.0, projecting 5 windows = 50.0
        let projected = forecast.project_total(5);
        assert!(
            (projected - 50.0).abs() < 1e-6,
            "Projection should be ~50.0, got {projected}"
        );
    }

    #[test]
    fn test_cost_forecast_display() {
        let costs = vec![10.0, 20.0];
        let forecast = CostForecast::from_window_costs(&costs);
        let display = format!("{forecast}");
        assert!(
            display.contains("slope="),
            "Should contain slope: {display}"
        );
        assert!(
            display.contains("INCREASING"),
            "Should show trend: {display}"
        );
    }

    #[test]
    fn test_cost_forecast_empty() {
        let forecast = CostForecast::from_window_costs(&[]);
        assert!(forecast.slope.abs() < 1e-10);
        assert!(forecast.pct_change().abs() < 1e-10);
        assert_eq!(forecast.trend_label(), "STABLE");
    }

    #[test]
    fn test_deterministic_rng_reproducibility() {
        let mut rng1 = DeterministicRng::new(123);
        let mut rng2 = DeterministicRng::new(123);

        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    #[test]
    fn test_simulate_request_model_profiles() {
        let cost_model = CostModel {
            cost_per_ms: 0.001,
            cost_per_mb: 0.0001,
            cost_per_1k_tokens: 0.01,
        };

        let mut rng_small = DeterministicRng::new(42);
        let mut rng_large = DeterministicRng::new(42);

        let mut total_small = 0.0;
        let mut total_large = 0.0;
        let n = 100;

        for _ in 0..n {
            let small = simulate_request(0, 0, 0, &cost_model, &mut rng_small);
            let large = simulate_request(2, 0, 0, &cost_model, &mut rng_large);
            total_small += small.total_cost;
            total_large += large.total_cost;
        }

        assert!(
            total_large > total_small,
            "gpt-large should cost more than gpt-small: large={total_large}, small={total_small}"
        );
    }

    #[test]
    fn test_inference_record_display() {
        let record = InferenceRecord {
            model: "test-model".to_string(),
            client: "test-client".to_string(),
            latency_ms: 50.0,
            memory_mb: 512.0,
            tokens: 1000,
            timestamp: 0,
            total_cost: 0.123456,
        };
        let display = format!("{record}");
        assert!(
            display.contains("test-model"),
            "Should contain model: {display}"
        );
        assert!(
            display.contains("test-client"),
            "Should contain client: {display}"
        );
        assert!(display.contains("cost="), "Should contain cost: {display}");
    }

    #[test]
    fn test_ascii_cost_chart_not_empty() {
        let entries = vec![
            ("model_a".to_string(), 10.0),
            ("model_b".to_string(), 20.0),
            ("model_c".to_string(), 5.0),
        ];
        let chart = CostTracker::ascii_cost_chart(&entries);
        assert!(!chart.is_empty(), "Chart should produce output");
        assert!(chart.contains('#'), "Chart should contain bar characters");
        assert!(chart.contains("model_a"), "Chart should contain labels");
    }
}
