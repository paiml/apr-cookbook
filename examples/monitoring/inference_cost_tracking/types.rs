//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

pub const NUM_MODELS: usize = 3;
pub const MODEL_NAMES: [&str; NUM_MODELS] = ["gpt-small", "gpt-medium", "gpt-large"];
pub const NUM_CLIENTS: usize = 4;
pub const CLIENT_NAMES: [&str; NUM_CLIENTS] = ["acme-corp", "globex-inc", "initech", "umbrella-co"];
pub const NUM_WINDOWS: usize = 6;
pub const REQUESTS_PER_WINDOW: usize = 100;
pub const BAR_WIDTH: usize = 40;

pub struct DeterministicRng {
    pub state: u64,
}
impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut h = DefaultHasher::new();
        self.state.hash(&mut h);
        self.state = h.finish();
        self.state
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    pub cost_per_ms: f64,
    pub cost_per_mb: f64,
    pub cost_per_1k_tokens: f64,
}
impl CostModel {
    pub fn compute_cost(&self, latency_ms: f64, memory_mb: f64, tokens: u64) -> f64 {
        self.cost_per_ms * latency_ms
            + self.cost_per_mb * memory_mb
            + self.cost_per_1k_tokens * (tokens as f64 / 1000.0)
    }
    pub fn cost_breakdown(&self, latency_ms: f64, memory_mb: f64, tokens: u64) -> CostBreakdown {
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

#[derive(Debug, Clone, Copy)]
pub struct CostBreakdown {
    pub compute: f64,
    pub memory: f64,
    pub tokens: f64,
}
impl CostBreakdown {
    pub fn total(&self) -> f64 {
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

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferenceRecord {
    pub model: String,
    pub client: String,
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub tokens: u64,
    pub timestamp: u64,
    pub total_cost: f64,
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

pub struct CostTracker {
    pub records: Vec<InferenceRecord>,
}
impl CostTracker {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
    pub fn add(&mut self, record: InferenceRecord) {
        self.records.push(record);
    }
    pub fn count(&self) -> usize {
        self.records.len()
    }
    pub fn total_cost(&self) -> f64 {
        self.records.iter().map(|r| r.total_cost).sum()
    }
    pub fn mean_cost(&self) -> f64 {
        if self.records.is_empty() {
            0.0
        } else {
            self.total_cost() / self.records.len() as f64
        }
    }
    pub fn cost_per_prediction(&self) -> f64 {
        self.mean_cost()
    }

    pub fn aggregate_by(
        &self,
        key_fn: impl Fn(&InferenceRecord) -> &str,
    ) -> Vec<(String, usize, f64, f64)> {
        let mut acc: Vec<(String, usize, f64)> = Vec::new();
        for r in &self.records {
            let k = key_fn(r);
            if let Some(e) = acc.iter_mut().find(|(n, _, _)| n == k) {
                e.1 += 1;
                e.2 += r.total_cost;
            } else {
                acc.push((k.to_string(), 1, r.total_cost));
            }
        }
        acc.into_iter()
            .map(|(n, c, t)| {
                let m = if c > 0 { t / c as f64 } else { 0.0 };
                (n, c, t, m)
            })
            .collect()
    }
    pub fn cost_by_model(&self) -> Vec<(String, usize, f64, f64)> {
        self.aggregate_by(|r| &r.model)
    }
    pub fn cost_by_client(&self) -> Vec<(String, usize, f64, f64)> {
        self.aggregate_by(|r| &r.client)
    }

    pub fn ascii_cost_chart(entries: &[(String, f64)]) -> String {
        let max = entries
            .iter()
            .map(|(_, c)| *c)
            .fold(0.0_f64, f64::max)
            .max(f64::EPSILON);
        let mut out = String::new();
        for (label, cost) in entries {
            let bar_len = ((*cost / max) * BAR_WIDTH as f64) as usize;
            out.push_str(&format!(
                "   {:>14} |{:<width$}| ${:.4}\n",
                label,
                "#".repeat(bar_len),
                cost,
                width = BAR_WIDTH
            ));
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertLevel {
    Ok,
    Warn,
    Critical,
    Exceeded,
}
impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Ok => "OK",
                Self::Warn => "WARN",
                Self::Critical => "CRITICAL",
                Self::Exceeded => "EXCEEDED",
            }
        )
    }
}

pub struct BudgetMonitor {
    pub budget: f64,
    pub warn_threshold: f64,
    pub critical_threshold: f64,
    pub spent: f64,
}
impl BudgetMonitor {
    pub fn new(budget: f64, warn_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            budget,
            warn_threshold,
            critical_threshold,
            spent: 0.0,
        }
    }
    pub fn record_spend(&mut self, amount: f64) {
        self.spent += amount;
    }
    #[allow(dead_code)]
    pub fn spent(&self) -> f64 {
        self.spent
    }
    pub fn utilization(&self) -> f64 {
        if self.budget <= 0.0 {
            if self.spent > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            self.spent / self.budget
        }
    }
    pub fn remaining(&self) -> f64 {
        self.budget - self.spent
    }
    pub fn alert_level(&self) -> AlertLevel {
        let u = self.utilization();
        if u >= 1.0 {
            AlertLevel::Exceeded
        } else if u >= self.critical_threshold {
            AlertLevel::Critical
        } else if u >= self.warn_threshold {
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

#[derive(Debug, Clone)]
pub struct CostForecast {
    pub window_costs: Vec<f64>,
    pub slope: f64,
    pub intercept: f64,
}
impl CostForecast {
    pub fn from_window_costs(costs: &[f64]) -> Self {
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
    pub fn predict(&self, window_idx: usize) -> f64 {
        self.intercept + self.slope * window_idx as f64
    }
    pub fn project_total(&self, num_windows: usize) -> f64 {
        (0..num_windows).map(|i| self.predict(i).max(0.0)).sum()
    }
    pub fn trend_label(&self) -> &str {
        if self.slope > 1e-9 {
            "INCREASING"
        } else if self.slope < -1e-9 {
            "DECREASING"
        } else {
            "STABLE"
        }
    }
    pub fn pct_change(&self) -> f64 {
        if self.window_costs.len() < 2 {
            return 0.0;
        }
        let first = self.window_costs[0];
        if first.abs() < f64::EPSILON {
            return 0.0;
        }
        (self.window_costs[self.window_costs.len() - 1] - first) / first * 100.0
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

pub fn linear_regression(values: &[f64]) -> (f64, f64) {
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
    (slope, (sum_y - slope * sum_x) / n)
}

pub fn simulate_request(
    model_idx: usize,
    client_idx: usize,
    timestamp: u64,
    cm: &CostModel,
    rng: &mut DeterministicRng,
) -> InferenceRecord {
    let (bl, bm, bt) = match model_idx {
        0 => (15.0, 256.0, 200.0),
        1 => (45.0, 1024.0, 500.0),
        _ => (120.0, 4096.0, 1500.0),
    };
    let (lat, mem, tok) = (
        bl * (0.7 + rng.next_f64() * 0.6),
        bm * (0.8 + rng.next_f64() * 0.4),
        (bt * (0.7 + rng.next_f64() * 0.6)) as u64,
    );
    InferenceRecord {
        model: MODEL_NAMES[model_idx].to_string(),
        client: CLIENT_NAMES[client_idx].to_string(),
        latency_ms: lat,
        memory_mb: mem,
        tokens: tok,
        timestamp,
        total_cost: cm.compute_cost(lat, mem, tok),
    }
}
