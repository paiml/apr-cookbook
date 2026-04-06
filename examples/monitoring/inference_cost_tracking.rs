//! Inference Cost Tracking and Resource Monitoring
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Per-request cost accounting, budget alerting, and cost trend analysis
//! for production inference workloads. Zero external dependencies beyond `std`.
//!
//! ## QA: Build, test, clippy, fmt PASS. IIUR compliant.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

const NUM_MODELS: usize = 3;
const MODEL_NAMES: [&str; NUM_MODELS] = ["gpt-small", "gpt-medium", "gpt-large"];
const NUM_CLIENTS: usize = 4;
const CLIENT_NAMES: [&str; NUM_CLIENTS] = ["acme-corp", "globex-inc", "initech", "umbrella-co"];
const NUM_WINDOWS: usize = 6;
const REQUESTS_PER_WINDOW: usize = 100;
const BAR_WIDTH: usize = 40;

struct DeterministicRng {
    state: u64,
}
impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        let mut h = DefaultHasher::new();
        self.state.hash(&mut h);
        self.state = h.finish();
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

#[derive(Debug, Clone, Copy)]
struct CostModel {
    cost_per_ms: f64,
    cost_per_mb: f64,
    cost_per_1k_tokens: f64,
}
impl CostModel {
    fn compute_cost(&self, latency_ms: f64, memory_mb: f64, tokens: u64) -> f64 {
        self.cost_per_ms * latency_ms
            + self.cost_per_mb * memory_mb
            + self.cost_per_1k_tokens * (tokens as f64 / 1000.0)
    }
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

#[derive(Debug, Clone, Copy)]
struct CostBreakdown {
    compute: f64,
    memory: f64,
    tokens: f64,
}
impl CostBreakdown {
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

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct InferenceRecord {
    model: String,
    client: String,
    latency_ms: f64,
    memory_mb: f64,
    tokens: u64,
    timestamp: u64,
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

struct CostTracker {
    records: Vec<InferenceRecord>,
}
impl CostTracker {
    fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
    fn add(&mut self, record: InferenceRecord) {
        self.records.push(record);
    }
    fn count(&self) -> usize {
        self.records.len()
    }
    fn total_cost(&self) -> f64 {
        self.records.iter().map(|r| r.total_cost).sum()
    }
    fn mean_cost(&self) -> f64 {
        if self.records.is_empty() {
            0.0
        } else {
            self.total_cost() / self.records.len() as f64
        }
    }
    fn cost_per_prediction(&self) -> f64 {
        self.mean_cost()
    }

    fn aggregate_by(
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
    fn cost_by_model(&self) -> Vec<(String, usize, f64, f64)> {
        self.aggregate_by(|r| &r.model)
    }
    fn cost_by_client(&self) -> Vec<(String, usize, f64, f64)> {
        self.aggregate_by(|r| &r.client)
    }

    fn ascii_cost_chart(entries: &[(String, f64)]) -> String {
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
enum AlertLevel {
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

struct BudgetMonitor {
    budget: f64,
    warn_threshold: f64,
    critical_threshold: f64,
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
    fn record_spend(&mut self, amount: f64) {
        self.spent += amount;
    }
    #[allow(dead_code)]
    fn spent(&self) -> f64 {
        self.spent
    }
    fn utilization(&self) -> f64 {
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
    fn remaining(&self) -> f64 {
        self.budget - self.spent
    }
    fn alert_level(&self) -> AlertLevel {
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
struct CostForecast {
    window_costs: Vec<f64>,
    slope: f64,
    intercept: f64,
}
impl CostForecast {
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
    fn predict(&self, window_idx: usize) -> f64 {
        self.intercept + self.slope * window_idx as f64
    }
    fn project_total(&self, num_windows: usize) -> f64 {
        (0..num_windows).map(|i| self.predict(i).max(0.0)).sum()
    }
    fn trend_label(&self) -> &str {
        if self.slope > 1e-9 {
            "INCREASING"
        } else if self.slope < -1e-9 {
            "DECREASING"
        } else {
            "STABLE"
        }
    }
    fn pct_change(&self) -> f64 {
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
    (slope, (sum_y - slope * sum_x) / n)
}

fn simulate_request(
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

fn main() {
    println!("=== Inference Cost Tracking ===\n");
    let mut rng = DeterministicRng::new(42);
    let cm = CostModel {
        cost_per_ms: 0.000_002,
        cost_per_mb: 0.000_000_5,
        cost_per_1k_tokens: 0.000_3,
    };
    println!("1. Cost Model: {cm}");
    for (i, name) in MODEL_NAMES.iter().enumerate() {
        let r = simulate_request(i, 0, 0, &cm, &mut rng);
        println!(
            "   {name}: {}",
            cm.cost_breakdown(r.latency_ms, r.memory_mb, r.tokens)
        );
    }
    // Section 2: Track requests
    let mut tracker = CostTracker::new();
    for i in 0..600 {
        let (mi, ci) = (rng.next_usize(NUM_MODELS), rng.next_usize(NUM_CLIENTS));
        tracker.add(simulate_request(
            mi,
            ci,
            1_700_000_000 + (i as u64) * 60,
            &cm,
            &mut rng,
        ));
    }
    println!(
        "\n2. Tracked {} requests, total=${:.6}, mean=${:.6}, cpp=${:.6}",
        tracker.count(),
        tracker.total_cost(),
        tracker.mean_cost(),
        tracker.cost_per_prediction()
    );
    for r in tracker.records.iter().take(3) {
        println!("   {r}");
    }
    // Section 3-4: Aggregation
    for (title, data) in [
        ("3. Per-Model", tracker.cost_by_model()),
        ("4. Per-Client", tracker.cost_by_client()),
    ] {
        println!("\n{title}:");
        for (n, c, t, m) in &data {
            println!("   {:>14} {:>5} reqs ${:.6} (avg ${:.6})", n, c, t, m);
        }
        let chart: Vec<(String, f64)> = data.iter().map(|(n, _, t, _)| (n.clone(), *t)).collect();
        print!("{}", CostTracker::ascii_cost_chart(&chart));
    }
    // Section 5: Budget monitoring
    let monthly_budget = tracker.total_cost() * 0.90;
    let mut bm = BudgetMonitor::new(monthly_budget, 0.80, 0.95);
    println!("\n5. Budget: ${:.4}", monthly_budget);
    let qs = tracker.count() / 4;
    for (i, r) in tracker.records.iter().enumerate() {
        bm.record_spend(r.total_cost);
        if (i + 1) % qs == 0 || i == tracker.count() - 1 {
            println!(
                "   At {:.0}%: {bm}",
                (i + 1) as f64 / tracker.count() as f64 * 100.0
            );
        }
    }
    // Section 6: Trend analysis
    let mut wc: Vec<f64> = Vec::new();
    let mut rng2 = DeterministicRng::new(99);
    for w in 0..NUM_WINDOWS {
        let mult = 1.0 + (w as f64) * 0.15;
        let wt: f64 = (0..REQUESTS_PER_WINDOW)
            .map(|_| {
                let (mi, ci) = (rng2.next_usize(NUM_MODELS), rng2.next_usize(NUM_CLIENTS));
                simulate_request(mi, ci, 0, &cm, &mut rng2).total_cost * mult
            })
            .sum();
        wc.push(wt);
        println!("\n   Window {w}: ${wt:.6} ({mult:.2}x)");
    }
    let fc = CostForecast::from_window_costs(&wc);
    println!(
        "   Trend: {fc}, projected next=${:.6}, 30-window=${:.4}",
        fc.predict(NUM_WINDOWS),
        fc.project_total(30)
    );
    println!("\n=== Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_model_compute() {
        let m = CostModel {
            cost_per_ms: 0.001,
            cost_per_mb: 0.0001,
            cost_per_1k_tokens: 0.01,
        };
        let cost = m.compute_cost(100.0, 512.0, 2000);
        assert!((cost - (0.1 + 0.0512 + 0.02)).abs() < 1e-10);
        assert!(m.compute_cost(0.0, 0.0, 0).abs() < 1e-10);
    }
    #[test]
    fn test_cost_breakdown() {
        let m = CostModel {
            cost_per_ms: 0.002,
            cost_per_mb: 0.001,
            cost_per_1k_tokens: 0.05,
        };
        let bd = m.cost_breakdown(50.0, 256.0, 1000);
        assert!((bd.compute - 0.1).abs() < 1e-10);
        assert!((bd.memory - 0.256).abs() < 1e-10);
        assert!((bd.tokens - 0.05).abs() < 1e-10);
        assert!((bd.total() - 0.406).abs() < 1e-10);
    }
    #[test]
    fn test_tracker_empty_and_aggregation() {
        let t = CostTracker::new();
        assert_eq!(t.count(), 0);
        assert!(t.total_cost().abs() < f64::EPSILON);
        assert!(t.cost_by_model().is_empty());
        let mut t2 = CostTracker::new();
        for (model, client, cost) in [("m1", "c1", 1.0), ("m1", "c2", 2.0), ("m2", "c1", 3.0)] {
            t2.add(InferenceRecord {
                model: model.into(),
                client: client.into(),
                latency_ms: 10.0,
                memory_mb: 100.0,
                tokens: 500,
                timestamp: 0,
                total_cost: cost,
            });
        }
        assert_eq!(t2.count(), 3);
        assert!((t2.total_cost() - 6.0).abs() < 1e-10);
        assert!((t2.mean_cost() - 2.0).abs() < 1e-10);
        assert_eq!(t2.cost_by_model().len(), 2);
        assert_eq!(t2.cost_by_client().len(), 2);
    }
    #[test]
    fn test_tracker_by_model_values() {
        let mut t = CostTracker::new();
        for i in 0..5 {
            t.add(InferenceRecord {
                model: "m1".into(),
                client: "c1".into(),
                latency_ms: 10.0,
                memory_mb: 100.0,
                tokens: 500,
                timestamp: i,
                total_cost: 1.0,
            });
        }
        let bm = t.cost_by_model();
        assert_eq!(bm[0].1, 5);
        assert!((bm[0].2 - 5.0).abs() < 1e-10);
        assert!((bm[0].3 - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_budget_monitor_thresholds() {
        let mut m = BudgetMonitor::new(100.0, 0.80, 0.95);
        assert_eq!(m.alert_level(), AlertLevel::Ok);
        assert!((m.remaining() - 100.0).abs() < 1e-10);
        m.record_spend(79.0);
        assert_eq!(m.alert_level(), AlertLevel::Ok);
        m.record_spend(2.0);
        assert_eq!(m.alert_level(), AlertLevel::Warn);
        m.record_spend(15.0);
        assert_eq!(m.alert_level(), AlertLevel::Critical);
        m.record_spend(5.0);
        assert_eq!(m.alert_level(), AlertLevel::Exceeded);
    }
    #[test]
    fn test_budget_zero_and_ordering() {
        let mut m = BudgetMonitor::new(0.0, 0.80, 0.95);
        assert!(m.utilization().abs() < f64::EPSILON);
        m.record_spend(1.0);
        assert!(m.utilization().is_infinite());
        assert!(
            AlertLevel::Ok < AlertLevel::Warn
                && AlertLevel::Warn < AlertLevel::Critical
                && AlertLevel::Critical < AlertLevel::Exceeded
        );
    }
    #[test]
    fn test_linear_regression() {
        let (s, i) = linear_regression(&[5.0, 5.0, 5.0, 5.0]);
        assert!(s.abs() < 1e-10 && (i - 5.0).abs() < 1e-10);
        let (s2, i2) = linear_regression(&[1.0, 3.0, 5.0, 7.0]);
        assert!((s2 - 2.0).abs() < 1e-10 && (i2 - 1.0).abs() < 1e-10);
        let (s3, i3) = linear_regression(&[42.0]);
        assert!(s3.abs() < 1e-10 && (i3 - 42.0).abs() < 1e-10);
    }
    #[test]
    fn test_cost_forecast() {
        let f1 = CostForecast::from_window_costs(&[10.0, 10.0, 10.0, 10.0]);
        assert_eq!(f1.trend_label(), "STABLE");
        assert!(f1.pct_change().abs() < 1e-10);
        assert!((f1.project_total(5) - 50.0).abs() < 1e-6);
        let f2 = CostForecast::from_window_costs(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(f2.trend_label(), "INCREASING");
        assert!((f2.predict(4) - 50.0).abs() < 1e-10);
        let f3 = CostForecast::from_window_costs(&[40.0, 30.0, 20.0, 10.0]);
        assert_eq!(f3.trend_label(), "DECREASING");
        let f4 = CostForecast::from_window_costs(&[]);
        assert_eq!(f4.trend_label(), "STABLE");
    }
    #[test]
    fn test_rng_determinism() {
        let s1: Vec<u64> = {
            let mut r = DeterministicRng::new(123);
            (0..10).map(|_| r.next_u64()).collect()
        };
        let s2: Vec<u64> = {
            let mut r = DeterministicRng::new(123);
            (0..10).map(|_| r.next_u64()).collect()
        };
        assert_eq!(s1, s2);
    }
    #[test]
    fn test_model_profiles_cost_ordering() {
        let cm = CostModel {
            cost_per_ms: 0.001,
            cost_per_mb: 0.0001,
            cost_per_1k_tokens: 0.01,
        };
        let (mut ts, mut tl) = (0.0, 0.0);
        let (mut rs, mut rl) = (DeterministicRng::new(42), DeterministicRng::new(42));
        for _ in 0..100 {
            ts += simulate_request(0, 0, 0, &cm, &mut rs).total_cost;
            tl += simulate_request(2, 0, 0, &cm, &mut rl).total_cost;
        }
        assert!(tl > ts);
    }
    #[test]
    fn test_ascii_chart() {
        let chart = CostTracker::ascii_cost_chart(&[("a".into(), 10.0), ("b".into(), 20.0)]);
        assert!(!chart.is_empty() && chart.contains('#'));
    }
    #[test]
    fn test_display_impls() {
        let cm = CostModel {
            cost_per_ms: 0.000_002,
            cost_per_mb: 0.000_000_5,
            cost_per_1k_tokens: 0.000_3,
        };
        assert!(format!("{cm}").contains("compute="));
        let bd = CostBreakdown {
            compute: 0.1,
            memory: 0.2,
            tokens: 0.05,
        };
        assert!(format!("{bd}").contains("total="));
        let bm = BudgetMonitor::new(100.0, 0.80, 0.95);
        assert!(format!("{bm}").contains("OK"));
        let fc = CostForecast::from_window_costs(&[10.0, 20.0]);
        assert!(format!("{fc}").contains("INCREASING"));
    }
}
