#![allow(unused_imports)]
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
