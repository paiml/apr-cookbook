//! # Recipe: Runs List — Filter + Sort by Status
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr runs list --status failed --sort duration --desc`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example runs_filter_sort` exits 0
//! 2. [x] `cargo test --example runs_filter_sort` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr runs list --status --sort --desc` in-process
//! 10. [x] Unit tests cover status filter, sort direction, tie-breakers
//!
//! ## Learning Objective
//! Demonstrates the `apr runs` subcommand's list/filter/sort semantics:
//! given a log of past training runs, filter by status (failed / succeeded /
//! running) and sort by a chosen field (duration, loss) with a stable
//! tie-breaker on run-id.
//!
//! ## Run Command
//! ```bash
//! cargo run --example runs_filter_sort
//! ```
//!
//! ## References
//! - Chen, A. et al. (2020). *Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle*. DEEM Workshop. arXiv:2003.04259

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Succeeded,
    Failed,
    Running,
}

impl RunStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Running => "running",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Run {
    pub id: String,
    pub status: RunStatus,
    pub duration_sec: u64,
    pub final_loss: f64,
}

pub fn fixture() -> Vec<Run> {
    vec![
        Run {
            id: "run-001".into(),
            status: RunStatus::Succeeded,
            duration_sec: 3_600,
            final_loss: 0.123,
        },
        Run {
            id: "run-002".into(),
            status: RunStatus::Failed,
            duration_sec: 240,
            final_loss: f64::NAN,
        },
        Run {
            id: "run-003".into(),
            status: RunStatus::Succeeded,
            duration_sec: 7_200,
            final_loss: 0.089,
        },
        Run {
            id: "run-004".into(),
            status: RunStatus::Failed,
            duration_sec: 1_800,
            final_loss: f64::INFINITY,
        },
        Run {
            id: "run-005".into(),
            status: RunStatus::Running,
            duration_sec: 600,
            final_loss: 0.412,
        },
        Run {
            id: "run-006".into(),
            status: RunStatus::Failed,
            duration_sec: 480,
            final_loss: f64::NAN,
        },
    ]
}

pub fn filter_by_status(runs: &[Run], status: RunStatus) -> Vec<Run> {
    runs.iter()
        .filter(|r| r.status == status)
        .cloned()
        .collect()
}

pub fn sort_by_duration_desc(runs: &mut [Run]) {
    runs.sort_by(|a, b| {
        b.duration_sec
            .cmp(&a.duration_sec)
            .then_with(|| a.id.cmp(&b.id))
    });
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("runs_filter_sort")?;
    println!("=== Recipe: {} ===", ctx.name());

    let all = fixture();
    let mut failed = filter_by_status(&all, RunStatus::Failed);
    sort_by_duration_desc(&mut failed);

    println!("Total runs:  {}", all.len());
    println!("Filter:      status=failed");
    println!("Sort:        duration DESC, id ASC\n");
    println!(
        "{:<10} {:<10} {:<10} {:<10}",
        "ID", "STATUS", "DUR(s)", "LOSS"
    );
    for r in &failed {
        let loss = if r.final_loss.is_finite() {
            format!("{:.3}", r.final_loss)
        } else {
            "N/A".into()
        };
        println!(
            "{:<10} {:<10} {:<10} {:<10}",
            r.id,
            r.status.label(),
            r.duration_sec,
            loss
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "n_total": all.len(),
        "n_filtered": failed.len(),
        "status_filter": "failed",
        "sort": "duration_desc",
        "runs": failed.iter().map(|r| json!({
            "id": r.id,
            "status": r.status.label(),
            "duration_sec": r.duration_sec,
            "final_loss": if r.final_loss.is_finite() { r.final_loss } else { 0.0 },
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("runs-filter.json");
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
    fn filter_picks_only_failed() {
        let f = filter_by_status(&fixture(), RunStatus::Failed);
        assert!(f.iter().all(|r| r.status == RunStatus::Failed));
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn filter_picks_only_succeeded() {
        let f = filter_by_status(&fixture(), RunStatus::Succeeded);
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn sort_desc_duration() {
        let mut runs = fixture();
        sort_by_duration_desc(&mut runs);
        for w in runs.windows(2) {
            assert!(w[0].duration_sec >= w[1].duration_sec);
        }
    }

    #[test]
    fn sort_tie_breaks_on_id() {
        let mut runs = vec![
            Run {
                id: "b-run".into(),
                status: RunStatus::Running,
                duration_sec: 100,
                final_loss: 0.0,
            },
            Run {
                id: "a-run".into(),
                status: RunStatus::Running,
                duration_sec: 100,
                final_loss: 0.0,
            },
        ];
        sort_by_duration_desc(&mut runs);
        assert_eq!(runs[0].id, "a-run");
    }

    #[test]
    fn status_label_roundtrip() {
        assert_eq!(RunStatus::Succeeded.label(), "succeeded");
        assert_eq!(RunStatus::Failed.label(), "failed");
        assert_eq!(RunStatus::Running.label(), "running");
    }
}
