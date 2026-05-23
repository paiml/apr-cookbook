//! # TUI Multi-Bar Progress
//!
//! Render a stack of progress bars for parallel tasks. Each task has
//! (name, completed, total). Returns the per-row tuple of (name,
//! ratio, status).
//!
//! Demonstrates the **TUI.49** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: indicatif MultiProgress widget pattern.
//!
//! Run with: cargo run --example tui_progress_multi_bar
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarStatus {
    Pending,
    InProgress,
    Complete,
}

#[derive(Debug, PartialEq)]
pub enum MultiBarVerdict {
    Ok {
        rows: Vec<(String, f64, BarStatus)>,
        all_complete: bool,
    },
    EmptyTasks,
}

pub fn render(tasks: &[(&str, u64, u64)]) -> MultiBarVerdict {
    if tasks.is_empty() {
        return MultiBarVerdict::EmptyTasks;
    }
    let mut rows = Vec::with_capacity(tasks.len());
    let mut all_complete = true;
    for (name, completed, total) in tasks {
        let (ratio, status) = if *total == 0 || *completed == 0 {
            (0.0, BarStatus::Pending)
        } else if *completed >= *total {
            (1.0, BarStatus::Complete)
        } else {
            (*completed as f64 / *total as f64, BarStatus::InProgress)
        };
        if status != BarStatus::Complete {
            all_complete = false;
        }
        rows.push(((*name).to_string(), ratio, status));
    }
    MultiBarVerdict::Ok { rows, all_complete }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_multi_bar")?;

    let tasks = [
        ("download", 50, 100),
        ("compile", 100, 100),
        ("test", 0, 50),
    ];
    println!("typical: {:?}", render(&tasks));

    let all_done = [("a", 10, 10)];
    println!("all done: {:?}", render(&all_done));
    println!("empty: {:?}", render(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_complete_recognized() {
        let v = render(&[("a", 10, 10), ("b", 5, 5)]);
        if let MultiBarVerdict::Ok { all_complete, .. } = v {
            assert!(all_complete);
        }
    }

    #[test]
    fn one_incomplete_means_not_all_complete() {
        let v = render(&[("a", 10, 10), ("b", 3, 10)]);
        if let MultiBarVerdict::Ok { all_complete, .. } = v {
            assert!(!all_complete);
        }
    }

    #[test]
    fn pending_when_completed_zero() {
        let v = render(&[("a", 0, 10)]);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert_eq!(rows[0].2, BarStatus::Pending);
        }
    }

    #[test]
    fn complete_status_when_done() {
        let v = render(&[("a", 5, 5)]);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert_eq!(rows[0].2, BarStatus::Complete);
        }
    }

    #[test]
    fn ratio_correct() {
        let v = render(&[("a", 50, 100)]);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert!((rows[0].1 - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn over_total_clamped_to_one() {
        let v = render(&[("a", 200, 100)]);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert!((rows[0].1 - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_total_pending() {
        let v = render(&[("a", 5, 0)]);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert_eq!(rows[0].2, BarStatus::Pending);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(render(&[]), MultiBarVerdict::EmptyTasks);
    }

    #[test]
    fn many_tasks() {
        let tasks: Vec<(&str, u64, u64)> = (0..50).map(|_| ("x", 5, 10)).collect();
        let v = render(&tasks);
        if let MultiBarVerdict::Ok { rows, .. } = v {
            assert_eq!(rows.len(), 50);
        }
    }

    #[test]
    fn deterministic() {
        let tasks = [("a", 5, 10)];
        let a = render(&tasks);
        let b = render(&tasks);
        assert_eq!(a, b);
    }
}
