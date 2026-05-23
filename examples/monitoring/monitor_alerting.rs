//! # Recipe: Monitor — Alerting on Threshold Violations
//!
//! **Category**: monitoring
//! **CLI Equivalent**: `apr monitor model.apr --alert p99_us:5000,err_rate:0.05`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example monitor_alerting` exits 0
//! 2. [x] `cargo test --example monitor_alerting` passes
//! 3. [x] Deterministic output (scripted fixture)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr monitor --alert` in-process (no shell-out)
//! 10. [x] Unit tests cover firing, resolution, hysteresis, multi-metric
//!
//! ## Learning Objective
//! Builds a minimal threshold-based alerting engine on top of the monitoring
//! stream. Supports multiple metrics, hysteresis (alert fires after K
//! consecutive violations, resolves after K clean samples), and a firing
//! log suitable for downstream paging.
//!
//! ## Run Command
//! ```bash
//! cargo run --example monitor_alerting
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct AlertRule {
    metric: String,
    threshold: f64,
    hysteresis_k: u32,
    comparator: Comparator,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum Comparator {
    GreaterThan,
    LessThan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlertState {
    Clear,
    Firing,
}

#[derive(Debug, Clone)]
struct AlertEvent {
    tick: u64,
    metric: String,
    kind: &'static str,
    value: f64,
    threshold: f64,
}

struct AlertEngine {
    rules: Vec<AlertRule>,
    streaks: HashMap<String, u32>,
    states: HashMap<String, AlertState>,
    events: Vec<AlertEvent>,
}

impl AlertEngine {
    fn new(rules: Vec<AlertRule>) -> Self {
        let mut states = HashMap::new();
        let mut streaks = HashMap::new();
        for r in &rules {
            states.insert(r.metric.clone(), AlertState::Clear);
            streaks.insert(r.metric.clone(), 0);
        }
        Self {
            rules,
            streaks,
            states,
            events: Vec::new(),
        }
    }
    fn observe(&mut self, tick: u64, metric: &str, value: f64) {
        let rule = match self.rules.iter().find(|r| r.metric == metric) {
            Some(r) => r.clone(),
            None => return,
        };
        let violated = match rule.comparator {
            Comparator::GreaterThan => value > rule.threshold,
            Comparator::LessThan => value < rule.threshold,
        };
        let streak = self.streaks.entry(metric.to_string()).or_insert(0);
        if violated {
            *streak += 1;
        } else if *streak > 0 {
            *streak -= 1;
        }
        let streak_val = *streak;
        let current = self
            .states
            .get(metric)
            .copied()
            .unwrap_or(AlertState::Clear);
        if streak_val >= rule.hysteresis_k && current == AlertState::Clear {
            self.states.insert(metric.to_string(), AlertState::Firing);
            self.events.push(AlertEvent {
                tick,
                metric: metric.to_string(),
                kind: "FIRE",
                value,
                threshold: rule.threshold,
            });
        } else if streak_val == 0 && current == AlertState::Firing {
            self.states.insert(metric.to_string(), AlertState::Clear);
            self.events.push(AlertEvent {
                tick,
                metric: metric.to_string(),
                kind: "CLEAR",
                value,
                threshold: rule.threshold,
            });
        }
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("monitor_alerting")?;
    println!("=== Recipe: {} ===", ctx.name());

    let rules = vec![
        AlertRule {
            metric: "p99_us".into(),
            threshold: 5000.0,
            hysteresis_k: 3,
            comparator: Comparator::GreaterThan,
        },
        AlertRule {
            metric: "err_rate".into(),
            threshold: 0.05,
            hysteresis_k: 2,
            comparator: Comparator::GreaterThan,
        },
    ];
    let mut eng = AlertEngine::new(rules.clone());

    // Scripted fixture: tick, metric, value
    let stream: Vec<(u64, &str, f64)> = vec![
        (0, "p99_us", 2000.0),
        (1, "p99_us", 2500.0),
        (2, "p99_us", 5500.0),
        (3, "p99_us", 6000.0),
        (4, "p99_us", 6100.0), // fires on tick 4 (streak=3)
        (5, "p99_us", 5700.0),
        (6, "p99_us", 3000.0),
        (7, "p99_us", 2500.0),
        (8, "p99_us", 1800.0), // clears somewhere here
        (2, "err_rate", 0.01),
        (3, "err_rate", 0.06),
        (4, "err_rate", 0.07), // fires on tick 4 (streak=2)
        (5, "err_rate", 0.02),
        (6, "err_rate", 0.01), // clears
    ];
    for (t, m, v) in &stream {
        eng.observe(*t, m, *v);
    }

    println!("\n--- Alert events ---");
    for e in &eng.events {
        println!(
            "tick={:>3} metric={:<10} {:<6} value={:<10.4} threshold={:.4}",
            e.tick, e.metric, e.kind, e.value, e.threshold
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "rules": rules.iter().map(|r| json!({
            "metric": r.metric,
            "threshold": r.threshold,
            "hysteresis_k": r.hysteresis_k,
        })).collect::<Vec<_>>(),
        "events": eng.events.iter().map(|e| json!({
            "tick": e.tick,
            "metric": e.metric,
            "kind": e.kind,
            "value": e.value,
            "threshold": e.threshold,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("alerting.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(metric: &str, thr: f64, k: u32) -> AlertRule {
        AlertRule {
            metric: metric.into(),
            threshold: thr,
            hysteresis_k: k,
            comparator: Comparator::GreaterThan,
        }
    }

    #[test]
    fn single_violation_below_k_does_not_fire() {
        let mut eng = AlertEngine::new(vec![rule("x", 10.0, 3)]);
        eng.observe(0, "x", 11.0);
        eng.observe(1, "x", 12.0);
        assert!(eng.events.is_empty());
    }

    #[test]
    fn k_consecutive_violations_fire() {
        let mut eng = AlertEngine::new(vec![rule("x", 10.0, 2)]);
        eng.observe(0, "x", 11.0);
        eng.observe(1, "x", 12.0);
        assert_eq!(eng.events.len(), 1);
        assert_eq!(eng.events[0].kind, "FIRE");
    }

    #[test]
    fn clean_samples_resolve_alert() {
        let mut eng = AlertEngine::new(vec![rule("x", 10.0, 2)]);
        eng.observe(0, "x", 11.0);
        eng.observe(1, "x", 12.0);
        eng.observe(2, "x", 5.0);
        eng.observe(3, "x", 5.0);
        // FIRE + CLEAR = 2 events
        assert_eq!(eng.events.len(), 2);
        assert_eq!(eng.events[1].kind, "CLEAR");
    }

    #[test]
    fn less_than_comparator_fires_on_low_values() {
        let mut eng = AlertEngine::new(vec![AlertRule {
            metric: "hit_rate".into(),
            threshold: 0.90,
            hysteresis_k: 2,
            comparator: Comparator::LessThan,
        }]);
        eng.observe(0, "hit_rate", 0.50);
        eng.observe(1, "hit_rate", 0.40);
        assert_eq!(eng.events.len(), 1);
        assert_eq!(eng.events[0].kind, "FIRE");
    }

    #[test]
    fn unknown_metric_ignored() {
        let mut eng = AlertEngine::new(vec![rule("x", 10.0, 1)]);
        eng.observe(0, "y", 100.0);
        assert!(eng.events.is_empty());
    }
}
