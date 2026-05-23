//! # apr monitor — Metrics-Frame NDJSON Envelope
//!
//! `apr monitor --json` emits one JSON object per line. Each frame must
//! include `step`, `loss`, `lr`, `tps`, and `wallclock_seconds` so
//! downstream consumers can plot any of them with no extra schema knowledge.
//! This recipe builds the envelope and asserts the contract: every frame
//! has the required keys, NaN values render as `null`, and the lines
//! parse back as valid JSON objects.
//!
//! Demonstrates the **MONITOR.6** recipe for PMAT-101 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MONITOR-003 + NDJSON 1.0 (ndjson.org)
//!
//! Run with: cargo run --example cli_monitor_metrics_envelope
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct MetricFrame {
    pub step: u64,
    pub loss: f64,
    pub lr: f64,
    pub tps: f64,
    pub wallclock_seconds: f64,
}

pub fn frame_to_json(f: &MetricFrame) -> Value {
    json!({
        "step": f.step,
        "loss": finite_or_null(f.loss),
        "lr": finite_or_null(f.lr),
        "tps": finite_or_null(f.tps),
        "wallclock_seconds": finite_or_null(f.wallclock_seconds),
    })
}

fn finite_or_null(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

pub fn render_ndjson(frames: &[MetricFrame]) -> String {
    frames
        .iter()
        .map(|f| serde_json::to_string(&frame_to_json(f)).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_metrics_envelope")?;

    let frames = vec![
        MetricFrame {
            step: 0,
            loss: 5.4,
            lr: 1e-4,
            tps: 12.0,
            wallclock_seconds: 0.5,
        },
        MetricFrame {
            step: 1,
            loss: 4.9,
            lr: 1e-4,
            tps: 13.2,
            wallclock_seconds: 1.0,
        },
        MetricFrame {
            step: 2,
            loss: f64::NAN,
            lr: 1e-4,
            tps: 11.5,
            wallclock_seconds: 1.5,
        },
    ];
    println!("{}", render_ndjson(&frames));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> MetricFrame {
        MetricFrame {
            step: 1,
            loss: 4.5,
            lr: 1e-4,
            tps: 12.0,
            wallclock_seconds: 0.5,
        }
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn frame_carries_all_required_keys() {
        let v = frame_to_json(&sample_frame());
        for key in ["step", "loss", "lr", "tps", "wallclock_seconds"] {
            assert!(v.get(key).is_some(), "missing key {key}");
        }
    }

    #[test]
    fn nan_metric_renders_as_null() {
        // CRITICAL: serde_json refuses to serialize NaN — must convert to null
        // so downstream consumers don't crash.
        let mut f = sample_frame();
        f.loss = f64::NAN;
        let v = frame_to_json(&f);
        assert_eq!(v["loss"], Value::Null);
    }

    #[test]
    fn inf_metric_renders_as_null() {
        let mut f = sample_frame();
        f.loss = f64::INFINITY;
        let v = frame_to_json(&f);
        assert_eq!(v["loss"], Value::Null);
    }

    #[test]
    fn ndjson_lines_parse_back_as_json() {
        let frames = vec![
            sample_frame(),
            MetricFrame {
                step: 2,
                loss: 4.0,
                lr: 1e-4,
                tps: 12.0,
                wallclock_seconds: 1.0,
            },
        ];
        let out = render_ndjson(&frames);
        for line in out.lines() {
            let parsed: Value = serde_json::from_str(line).unwrap();
            assert!(parsed.is_object());
        }
    }

    #[test]
    fn empty_frames_yield_empty_output() {
        assert_eq!(render_ndjson(&[]), "");
    }

    #[test]
    fn step_field_is_unsigned_integer() {
        // Important: step must serialize as JSON integer, not float.
        let v = frame_to_json(&sample_frame());
        assert_eq!(v["step"], json!(1u64));
    }
}
