//! # Recipe: TUI Health Dashboard
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr tui --view health`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example tui_health_dashboard` exits 0
//! 2. [x] `cargo test --example tui_health_dashboard` passes
//! 3. [x] Deterministic output (fixed metrics)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr tui --view health` render in-process
//! 10. [x] Unit tests cover verdict thresholds, box rendering, bar widths
//!
//! ## Learning Objective
//! Demonstrates a minimal ASCII TUI: render a health dashboard with metric
//! boxes (CPU, memory, latency, QPS) and a top-level verdict (HEALTHY /
//! DEGRADED / UNHEALTHY) based on threshold logic. The renderer is
//! byte-deterministic so snapshot tests are trivial.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tui_health_dashboard
//! ```
//!
//! ## References
//! - Rosenfeld, L. & Morville, P. (2006). *Information Architecture for the World Wide Web* (3rd ed.). O'Reilly Media.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct Metrics {
    pub cpu_pct: u32,
    pub mem_pct: u32,
    pub latency_p95_ms: u32,
    pub qps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Healthy,
    Degraded,
    Unhealthy,
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Degraded => "DEGRADED",
            Self::Unhealthy => "UNHEALTHY",
        }
    }
}

pub fn score(m: &Metrics) -> Verdict {
    let bad_cpu = m.cpu_pct > 90;
    let bad_mem = m.mem_pct > 90;
    let bad_lat = m.latency_p95_ms > 500;
    let any_bad = bad_cpu || bad_mem || bad_lat;

    let warn_cpu = m.cpu_pct > 70;
    let warn_mem = m.mem_pct > 75;
    let warn_lat = m.latency_p95_ms > 200;
    let any_warn = warn_cpu || warn_mem || warn_lat;

    match (any_bad, any_warn) {
        (true, _) => Verdict::Unhealthy,
        (_, true) => Verdict::Degraded,
        _ => Verdict::Healthy,
    }
}

/// Return a width-20 bar for a 0-100 percentage.
pub fn bar_for_pct(pct: u32) -> String {
    let pct = pct.min(100);
    let filled = (pct / 5) as usize;
    let empty = 20 - filled;
    format!("[{}{}]", "#".repeat(filled), " ".repeat(empty))
}

pub fn render_dashboard(m: &Metrics, v: Verdict) -> String {
    let mut s = String::new();
    s.push_str("+----------------------- HEALTH ------------------------+\n");
    s.push_str(&format!("| STATUS: {:<46}|\n", v.label()));
    s.push_str("+-------------------------------------------------------+\n");
    s.push_str(&format!(
        "| CPU:  {} {:>4}%                        |\n",
        bar_for_pct(m.cpu_pct),
        m.cpu_pct
    ));
    s.push_str(&format!(
        "| MEM:  {} {:>4}%                        |\n",
        bar_for_pct(m.mem_pct),
        m.mem_pct
    ));
    s.push_str(&format!(
        "| P95:  {:>6} ms                                         |\n",
        m.latency_p95_ms
    ));
    s.push_str(&format!(
        "| QPS:  {:>6}                                            |\n",
        m.qps
    ));
    s.push_str("+-------------------------------------------------------+\n");
    s
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("tui_health_dashboard")?;
    println!("=== Recipe: {} ===", ctx.name());

    let metrics = Metrics {
        cpu_pct: 62,
        mem_pct: 80,
        latency_p95_ms: 210,
        qps: 145,
    };
    let verdict = score(&metrics);
    let dashboard = render_dashboard(&metrics, verdict);
    println!("{}", dashboard);

    let report = json!({
        "recipe": ctx.name(),
        "verdict": verdict.label(),
        "metrics": {
            "cpu_pct": metrics.cpu_pct,
            "mem_pct": metrics.mem_pct,
            "latency_p95_ms": metrics.latency_p95_ms,
            "qps": metrics.qps,
        },
        "render_bytes": dashboard.len(),
    });
    let out = ctx.path("tui-health.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_when_all_low() {
        let m = Metrics {
            cpu_pct: 10,
            mem_pct: 20,
            latency_p95_ms: 30,
            qps: 100,
        };
        assert_eq!(score(&m), Verdict::Healthy);
    }

    #[test]
    fn degraded_on_warn_thresholds() {
        let m = Metrics {
            cpu_pct: 75,
            mem_pct: 60,
            latency_p95_ms: 220,
            qps: 100,
        };
        assert_eq!(score(&m), Verdict::Degraded);
    }

    #[test]
    fn unhealthy_on_error_thresholds() {
        let m = Metrics {
            cpu_pct: 95,
            mem_pct: 60,
            latency_p95_ms: 100,
            qps: 100,
        };
        assert_eq!(score(&m), Verdict::Unhealthy);
    }

    #[test]
    fn bar_length_is_stable() {
        assert_eq!(bar_for_pct(0), "[                    ]");
        assert_eq!(bar_for_pct(50), "[##########          ]");
        assert_eq!(bar_for_pct(100), "[####################]");
        assert_eq!(bar_for_pct(9999).len(), 22); // "[" + 20 + "]"
    }

    #[test]
    fn render_is_deterministic() {
        let m = Metrics {
            cpu_pct: 10,
            mem_pct: 20,
            latency_p95_ms: 30,
            qps: 100,
        };
        let a = render_dashboard(&m, score(&m));
        let b = render_dashboard(&m, score(&m));
        assert_eq!(a, b);
    }
}
