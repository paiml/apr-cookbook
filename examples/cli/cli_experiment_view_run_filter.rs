//! # apr experiment view — Run Filter Predicate
//!
//! `apr experiment view` opens an interactive TUI of all training runs.
//! This recipe builds the per-run filter predicate as a pure function
//! so a CI pipeline can preview which runs the TUI will show before
//! invoking the binary. Filter modes: `--status`, `--tag`, `--since`,
//! `--max-loss`. Multiple filters compose AND-style.
//!
//! Demonstrates the **EXPERIMENT.4** recipe for PMAT-102 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXPERIMENT-001
//!
//! Run with: cargo run --example cli_experiment_view_run_filter
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone)]
pub struct RunRecord {
    pub id: String,
    pub status: RunStatus,
    pub tags: Vec<String>,
    pub started_unix: u64,
    pub final_loss: Option<f64>,
}

#[derive(Debug, Default, Clone)]
pub struct RunFilter {
    pub status: Option<RunStatus>,
    pub tag: Option<String>,
    pub since_unix: Option<u64>,
    pub max_loss: Option<f64>,
}

pub fn matches(r: &RunRecord, f: &RunFilter) -> bool {
    if let Some(s) = f.status {
        if r.status != s {
            return false;
        }
    }
    if let Some(t) = &f.tag {
        if !r.tags.iter().any(|tag| tag == t) {
            return false;
        }
    }
    if let Some(since) = f.since_unix {
        if r.started_unix < since {
            return false;
        }
    }
    if let Some(max) = f.max_loss {
        match r.final_loss {
            Some(l) if l <= max => {}
            _ => return false,
        }
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_view_run_filter")?;

    let runs = vec![
        RunRecord {
            id: "run-001".into(),
            status: RunStatus::Completed,
            tags: vec!["bf16".into(), "qwen3".into()],
            started_unix: 1_715_000_000,
            final_loss: Some(2.4),
        },
        RunRecord {
            id: "run-002".into(),
            status: RunStatus::Failed,
            tags: vec!["bf16".into()],
            started_unix: 1_715_100_000,
            final_loss: None,
        },
        RunRecord {
            id: "run-003".into(),
            status: RunStatus::Running,
            tags: vec!["fp8".into()],
            started_unix: 1_715_200_000,
            final_loss: None,
        },
        RunRecord {
            id: "run-004".into(),
            status: RunStatus::Completed,
            tags: vec!["bf16".into()],
            started_unix: 1_715_300_000,
            final_loss: Some(3.1),
        },
    ];

    let f = RunFilter {
        status: Some(RunStatus::Completed),
        tag: Some("bf16".into()),
        max_loss: Some(2.5),
        ..Default::default()
    };

    println!("=== Filter: {f:?} ===");
    for r in &runs {
        println!("  {}  match={}", r.id, matches(r, &f));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_run() -> RunRecord {
        RunRecord {
            id: "r1".into(),
            status: RunStatus::Completed,
            tags: vec!["bf16".into(), "qwen3".into()],
            started_unix: 1_715_000_000,
            final_loss: Some(2.5),
        }
    }

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_filter_matches_everything() {
        assert!(matches(&sample_run(), &RunFilter::default()));
    }

    #[test]
    fn status_filter_excludes_others() {
        let f = RunFilter {
            status: Some(RunStatus::Failed),
            ..Default::default()
        };
        assert!(!matches(&sample_run(), &f));
    }

    #[test]
    fn tag_filter_requires_match() {
        let f = RunFilter {
            tag: Some("nonexistent".into()),
            ..Default::default()
        };
        assert!(!matches(&sample_run(), &f));
    }

    #[test]
    fn since_filter_excludes_older_runs() {
        let f = RunFilter {
            since_unix: Some(1_716_000_000),
            ..Default::default()
        };
        assert!(!matches(&sample_run(), &f));
    }

    #[test]
    fn max_loss_filter_excludes_high_loss() {
        let f = RunFilter {
            max_loss: Some(1.0),
            ..Default::default()
        };
        assert!(!matches(&sample_run(), &f));
    }

    #[test]
    fn max_loss_filter_excludes_runs_with_no_final_loss() {
        // Failed/running runs lack a final loss — must NOT pass max_loss filter.
        let mut r = sample_run();
        r.final_loss = None;
        let f = RunFilter {
            max_loss: Some(100.0),
            ..Default::default()
        };
        assert!(!matches(&r, &f));
    }

    #[test]
    fn filters_compose_and_style() {
        // status=Completed AND tag=bf16 AND max_loss<=3.0 — sample passes all.
        let f = RunFilter {
            status: Some(RunStatus::Completed),
            tag: Some("bf16".into()),
            max_loss: Some(3.0),
            ..Default::default()
        };
        assert!(matches(&sample_run(), &f));
    }
}
