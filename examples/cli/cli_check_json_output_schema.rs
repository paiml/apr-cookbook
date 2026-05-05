//! # apr check — `--json` Output Schema (CI-Parseable Report)
//!
//! `apr check --json <MODEL.apr>` emits a stable JSON report consumable by CI
//! systems (GitHub Actions annotations, Buildkite test summaries, etc.). The
//! schema is contract-stable: top-level fields `model`, `apr_version`,
//! `summary`, `stages` are guaranteed; per-stage objects always carry
//! `name`, `status`, and `detail` even when `status == "pass"` (CI parsers
//! can rely on the keys existing).
//!
//! This recipe builds a representative report and asserts the schema in
//! tests so a future CLI change is forced to update both the emitter and
//! the consumer.
//!
//! Demonstrates the **CHECK.3** recipe for PMAT-088 (apr check coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CHECK-003 + serde_json 1.x (RFC 8259)
//!
//! Run with: cargo run --example cli_check_json_output_schema
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Clone)]
struct StageReport {
    name: &'static str,
    status: &'static str,
    detail: &'static str,
}

fn build_report(model: &str, apr_version: u32, stages: &[StageReport]) -> Value {
    let total = stages.len();
    let passed = stages.iter().filter(|s| s.status == "pass").count();
    let failed = stages.iter().filter(|s| s.status == "fail").count();
    let skipped = stages.iter().filter(|s| s.status == "skipped").count();

    json!({
        "model": model,
        "apr_version": apr_version,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "ok": failed == 0,
        },
        "stages": stages.iter().map(|s| json!({
            "name": s.name,
            "status": s.status,
            "detail": s.detail,
        })).collect::<Vec<_>>()
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_check_json_output_schema")?;

    let stages = [
        StageReport {
            name: "magic",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "version",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "crc32",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "tensor-shape",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "tensor-dtype",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "quantization",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "tokenizer",
            status: "skipped",
            detail: "no tokenizer",
        },
        StageReport {
            name: "provenance",
            status: "pass",
            detail: "",
        },
        StageReport {
            name: "signature",
            status: "skipped",
            detail: "no signature",
        },
        StageReport {
            name: "contract",
            status: "pass",
            detail: "",
        },
    ];

    let report = build_report("model.apr", 2, &stages);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_runs() {
        main().expect("recipe execution failed");
    }

    fn sample_report() -> Value {
        let stages = [
            StageReport {
                name: "magic",
                status: "pass",
                detail: "",
            },
            StageReport {
                name: "version",
                status: "pass",
                detail: "",
            },
        ];
        build_report("m.apr", 2, &stages)
    }

    #[test]
    fn top_level_keys_present() {
        let r = sample_report();
        for key in ["model", "apr_version", "summary", "stages"] {
            assert!(r.get(key).is_some(), "missing top-level key {key}");
        }
    }

    #[test]
    fn summary_carries_ok_boolean() {
        // The `summary.ok` boolean is what CI gates check first; it MUST exist.
        let r = sample_report();
        assert_eq!(r["summary"]["ok"], json!(true));
    }

    #[test]
    fn fail_flips_summary_ok_false() {
        let stages = [
            StageReport {
                name: "magic",
                status: "pass",
                detail: "",
            },
            StageReport {
                name: "crc32",
                status: "fail",
                detail: "section CRC mismatch",
            },
        ];
        let r = build_report("m.apr", 2, &stages);
        assert_eq!(r["summary"]["ok"], json!(false));
        assert_eq!(r["summary"]["failed"], json!(1));
    }

    #[test]
    fn every_stage_has_name_status_detail_keys() {
        // Even passing stages keep the `detail` key (empty string) so CI
        // parsers don't have to special-case its absence.
        let r = sample_report();
        let stages = r["stages"].as_array().unwrap();
        for s in stages {
            for key in ["name", "status", "detail"] {
                assert!(s.get(key).is_some(), "stage missing key {key}: {s:?}");
            }
        }
    }

    #[test]
    fn status_uses_lowercase_enum_strings() {
        // pass / fail / skipped — CI grep depends on exact lowercase tokens.
        let stages = [
            StageReport {
                name: "a",
                status: "pass",
                detail: "",
            },
            StageReport {
                name: "b",
                status: "fail",
                detail: "x",
            },
            StageReport {
                name: "c",
                status: "skipped",
                detail: "y",
            },
        ];
        let r = build_report("m.apr", 2, &stages);
        let arr = r["stages"].as_array().unwrap();
        assert_eq!(arr[0]["status"], json!("pass"));
        assert_eq!(arr[1]["status"], json!("fail"));
        assert_eq!(arr[2]["status"], json!("skipped"));
    }
}
