//! # apr rosetta validate-stats — Per-Tensor Report Aggregator
//!
//! `apr rosetta validate-stats` emits a per-tensor report grouping
//! findings by severity (Pass / Warn / Fail) and tagging each with the
//! statistic that triggered the verdict. This recipe builds the
//! aggregator and asserts the priority ordering: Fail wins over Warn,
//! and the report is sorted by tensor name for deterministic CI logs.
//!
//! Demonstrates the **ROSETTA-VALIDATE.3** recipe for PMAT-097 (apr rosetta validate-stats coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-202
//!
//! Run with: cargo run --example cli_rosetta_validate_stats_per_tensor_report
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorReport {
    pub tensor: String,
    pub severity: Severity,
    pub trigger_stat: Option<&'static str>,
}

pub fn classify_severity(z_score: f64, warn_at: f64, fail_at: f64) -> Severity {
    let abs = z_score.abs();
    if abs >= fail_at {
        Severity::Fail
    } else if abs >= warn_at {
        Severity::Warn
    } else {
        Severity::Pass
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    pub mean_z: f64,
    pub std_z: f64,
    pub min_z: f64,
    pub max_z: f64,
}

pub fn build_report(name: &str, stats: TensorStats, warn_at: f64, fail_at: f64) -> TensorReport {
    let candidates = [
        ("mean", stats.mean_z),
        ("std", stats.std_z),
        ("min", stats.min_z),
        ("max", stats.max_z),
    ];
    let mut highest = Severity::Pass;
    let mut trigger: Option<&'static str> = None;
    for (label, z) in candidates {
        let s = classify_severity(z, warn_at, fail_at);
        if s > highest {
            highest = s;
            trigger = Some(label);
        }
    }
    TensorReport {
        tensor: name.into(),
        severity: highest,
        trigger_stat: trigger,
    }
}

pub fn aggregate_reports(reports: &mut [TensorReport]) {
    // Deterministic ordering: sort by tensor name (CI log diff-friendly).
    reports.sort_by(|a, b| a.tensor.cmp(&b.tensor));
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_validate_stats_per_tensor_report")?;

    let mut reports = vec![
        build_report(
            "lm_head",
            TensorStats {
                mean_z: 0.5,
                std_z: 1.0,
                min_z: -1.5,
                max_z: 1.8,
            },
            2.0,
            3.0,
        ),
        build_report(
            "embed_tokens",
            TensorStats {
                mean_z: 4.5,
                std_z: 1.0,
                min_z: -1.5,
                max_z: 1.8,
            },
            2.0,
            3.0,
        ),
        build_report(
            "layers.0.q_proj",
            TensorStats {
                mean_z: 1.0,
                std_z: 2.5,
                min_z: -1.0,
                max_z: 1.0,
            },
            2.0,
            3.0,
        ),
    ];
    aggregate_reports(&mut reports);

    println!("=== Per-tensor report ===");
    for r in &reports {
        println!(
            "  {:?} {} (trigger={:?})",
            r.severity, r.tensor, r.trigger_stat
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral_stats() -> TensorStats {
        TensorStats {
            mean_z: 0.5,
            std_z: 0.5,
            min_z: -0.5,
            max_z: 0.5,
        }
    }

    #[test]
    fn report_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn neutral_stats_yields_pass() {
        let r = build_report("x", neutral_stats(), 2.0, 3.0);
        assert_eq!(r.severity, Severity::Pass);
        assert_eq!(r.trigger_stat, None);
    }

    #[test]
    fn fail_severity_wins_over_warn() {
        // mean is Warn (z=2.5), std is Fail (z=4.0) → Fail must win.
        let stats = TensorStats {
            mean_z: 2.5,
            std_z: 4.0,
            min_z: -1.0,
            max_z: 1.0,
        };
        let r = build_report("x", stats, 2.0, 3.0);
        assert_eq!(r.severity, Severity::Fail);
        assert_eq!(r.trigger_stat, Some("std"));
    }

    #[test]
    fn negative_z_score_uses_abs() {
        // Min z of -3.5 is Fail when fail_at = 3.0.
        let stats = TensorStats {
            mean_z: 0.0,
            std_z: 0.0,
            min_z: -3.5,
            max_z: 0.0,
        };
        let r = build_report("x", stats, 2.0, 3.0);
        assert_eq!(r.severity, Severity::Fail);
        assert_eq!(r.trigger_stat, Some("min"));
    }

    #[test]
    fn aggregator_sorts_by_tensor_name() {
        let mut reports = vec![
            TensorReport {
                tensor: "z".into(),
                severity: Severity::Pass,
                trigger_stat: None,
            },
            TensorReport {
                tensor: "a".into(),
                severity: Severity::Pass,
                trigger_stat: None,
            },
            TensorReport {
                tensor: "m".into(),
                severity: Severity::Pass,
                trigger_stat: None,
            },
        ];
        aggregate_reports(&mut reports);
        let names: Vec<&str> = reports.iter().map(|r| r.tensor.as_str()).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn severity_ordering_is_pass_lt_warn_lt_fail() {
        assert!(Severity::Pass < Severity::Warn);
        assert!(Severity::Warn < Severity::Fail);
    }

    #[test]
    fn warn_severity_when_no_fail_present() {
        let stats = TensorStats {
            mean_z: 2.5, // Warn
            std_z: 0.0,
            min_z: 0.0,
            max_z: 0.0,
        };
        let r = build_report("x", stats, 2.0, 3.0);
        assert_eq!(r.severity, Severity::Warn);
        assert_eq!(r.trigger_stat, Some("mean"));
    }
}
