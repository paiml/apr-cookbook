//! # apr cbtop — `--headless --json` Output Schema
//!
//! `apr cbtop --headless --json --output report.json` runs the monitor
//! without a TUI and emits a stable JSON report. Top-level fields:
//! `model`, `iterations`, `warmup`, `metrics`, `verdict`. Per-iteration
//! `metrics` always carry `tps` and `brick_score` keys (even if zero) so
//! CI parsers don't have to special-case absence. This recipe pins the
//! schema so future regressions break the test.
//!
//! Demonstrates the **CBTOP.5** recipe for PMAT-094 (apr cbtop coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CBTOP-CI-001 + serde_json 1.x (RFC 8259)
//!
//! Run with: cargo run --example cli_cbtop_headless_json_schema
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub struct IterationMetric {
    pub tps: f64,
    pub brick_score: f64,
}

pub fn build_report(model: &str, warmup: u32, metrics: &[IterationMetric], verdict: &str) -> Value {
    json!({
        "model": model,
        "warmup": warmup,
        "iterations": metrics.len(),
        "metrics": metrics
            .iter()
            .enumerate()
            .map(|(i, m)| json!({
                "iter": i,
                "tps": m.tps,
                "brick_score": m.brick_score
            }))
            .collect::<Vec<_>>(),
        "verdict": verdict
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_cbtop_headless_json_schema")?;

    let metrics = (0..5)
        .map(|i| IterationMetric {
            tps: 80.0 + i as f64,
            brick_score: 90.0 + i as f64 * 0.5,
        })
        .collect::<Vec<_>>();
    let report = build_report("qwen2.5-coder-1.5b", 10, &metrics, "PASS");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> Value {
        let metrics = [
            IterationMetric {
                tps: 80.0,
                brick_score: 90.0,
            },
            IterationMetric {
                tps: 85.0,
                brick_score: 92.0,
            },
        ];
        build_report("model", 3, &metrics, "PASS")
    }

    #[test]
    fn schema_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn top_level_keys_present() {
        let r = sample_report();
        for key in ["model", "warmup", "iterations", "metrics", "verdict"] {
            assert!(r.get(key).is_some(), "missing top-level key {key}");
        }
    }

    #[test]
    fn iteration_count_matches_metrics_array_length() {
        let r = sample_report();
        let n = r["iterations"].as_u64().unwrap();
        let m = r["metrics"].as_array().unwrap().len() as u64;
        assert_eq!(n, m);
    }

    #[test]
    fn each_metric_has_tps_and_brick_score() {
        // CI parsers depend on the keys existing — pin the contract.
        let r = sample_report();
        for m in r["metrics"].as_array().unwrap() {
            assert!(m.get("tps").is_some());
            assert!(m.get("brick_score").is_some());
            assert!(m.get("iter").is_some());
        }
    }

    #[test]
    fn empty_metrics_array_is_allowed() {
        // Pre-warmup or skip-iteration runs can have zero measured iterations.
        let r = build_report("m", 10, &[], "SKIPPED");
        assert_eq!(r["iterations"], json!(0));
        assert!(r["metrics"].as_array().unwrap().is_empty());
    }

    #[test]
    fn verdict_is_uppercase_string() {
        // CI grep looks for /^"verdict":\s*"PASS"|"FAIL"|"SKIPPED"$/ — pin format.
        let r = sample_report();
        let v = r["verdict"].as_str().unwrap();
        assert_eq!(v, v.to_ascii_uppercase());
    }
}
