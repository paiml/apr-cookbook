//! # Recipe: Grid Search with Early Stopping
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr tune --strategy grid --patience 3`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example tune_grid_early_stop` exits 0
//! 2. [x] `cargo test --example tune_grid_early_stop` passes
//! 3. [x] Deterministic output (fixed grid)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr tune --strategy grid --patience` in-process
//! 10. [x] Unit tests cover patience counter, best-so-far update, early exit
//!
//! ## Learning Objective
//! Demonstrates classical grid search with early stopping: enumerate a
//! cartesian product of hyperparameter candidates, evaluate each, and stop
//! when the best score has not improved for `patience` consecutive trials.
//! Mirrors `apr tune --strategy grid --patience N`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tune_grid_early_stop
//! ```
//!
//! ## References
//! - Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR. URL: https://www.jmlr.org/papers/v13/bergstra12a.html

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPoint {
    pub lr: f64,
    pub dropout: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredTrial {
    pub lr: f64,
    pub dropout: f64,
    pub score: f64,
}

/// Objective: minimize |lr - 0.003| + |dropout - 0.25| + tiny sin wiggle.
pub fn objective(p: &GridPoint) -> f64 {
    (p.lr - 0.003).abs() + (p.dropout - 0.25).abs() + 0.01 * (p.lr * 1000.0).sin()
}

pub fn cartesian_grid(lrs: &[f64], dropouts: &[f64]) -> Vec<GridPoint> {
    let mut out = Vec::with_capacity(lrs.len() * dropouts.len());
    for lr in lrs {
        for dr in dropouts {
            out.push(GridPoint {
                lr: *lr,
                dropout: *dr,
            });
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct TuneRun {
    pub trials: Vec<ScoredTrial>,
    pub stopped_early: bool,
    pub n_evaluated: usize,
    pub best: Option<ScoredTrial>,
}

pub fn grid_search_with_patience(grid: &[GridPoint], patience: u32) -> TuneRun {
    let mut trials = Vec::new();
    let mut best: Option<ScoredTrial> = None;
    let mut no_improve: u32 = 0;
    let mut stopped_early = false;
    let mut n_evaluated = 0;

    for p in grid {
        let score = objective(p);
        let trial = ScoredTrial {
            lr: p.lr,
            dropout: p.dropout,
            score,
        };
        n_evaluated += 1;
        let improved = match best {
            None => true,
            Some(b) => score < b.score - 1e-9,
        };
        if improved {
            best = Some(trial);
            no_improve = 0;
        } else {
            no_improve += 1;
        }
        trials.push(trial);
        if no_improve >= patience {
            stopped_early = true;
            break;
        }
    }

    TuneRun {
        trials,
        stopped_early,
        n_evaluated,
        best,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("tune_grid_early_stop")?;
    println!("=== Recipe: {} ===", ctx.name());

    let lrs = vec![1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
    let dropouts = vec![0.0, 0.1, 0.25, 0.4];
    let grid = cartesian_grid(&lrs, &dropouts);
    let patience = 3u32;

    let run = grid_search_with_patience(&grid, patience);

    println!(
        "Grid size:   {}  ({} lrs x {} dropouts)",
        grid.len(),
        lrs.len(),
        dropouts.len()
    );
    println!("Patience:    {}", patience);
    println!("Evaluated:   {}/{}", run.n_evaluated, grid.len());
    println!("Stopped early: {}", run.stopped_early);
    if let Some(b) = run.best {
        println!(
            "Best: lr={:.4e} dropout={:.2} score={:.6}",
            b.lr, b.dropout, b.score
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "grid_size": grid.len(),
        "patience": patience,
        "n_evaluated": run.n_evaluated,
        "stopped_early": run.stopped_early,
        "best": run.best.map(|b| json!({
            "lr": b.lr,
            "dropout": b.dropout,
            "score": b.score,
        })),
        "trials": run.trials.iter().map(|t| json!({
            "lr": t.lr,
            "dropout": t.dropout,
            "score": t.score,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("tune-grid.json");
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
    fn cartesian_grid_size_is_product() {
        let g = cartesian_grid(&[1e-4, 1e-3], &[0.0, 0.1, 0.2]);
        assert_eq!(g.len(), 6);
    }

    #[test]
    fn best_is_captured() {
        let g = vec![
            GridPoint {
                lr: 0.003,
                dropout: 0.25,
            },
            GridPoint {
                lr: 1.0,
                dropout: 0.9,
            },
        ];
        let run = grid_search_with_patience(&g, 10);
        let best = run.best.expect("best");
        assert!((best.lr - 0.003).abs() < 1e-9);
    }

    #[test]
    fn patience_stops_when_no_improvement() {
        // Construct a grid where the first point is the best.
        let g = vec![
            GridPoint {
                lr: 0.003,
                dropout: 0.25,
            }, // best
            GridPoint {
                lr: 1.0,
                dropout: 0.9,
            }, // worse
            GridPoint {
                lr: 2.0,
                dropout: 0.9,
            }, // worse
            GridPoint {
                lr: 3.0,
                dropout: 0.9,
            }, // worse -> patience=3 triggers
            GridPoint {
                lr: 4.0,
                dropout: 0.9,
            }, // NOT visited
        ];
        let run = grid_search_with_patience(&g, 3);
        assert!(run.stopped_early);
        assert!(run.n_evaluated < g.len());
    }

    #[test]
    fn no_early_stop_when_always_improving() {
        // Ascending-quality grid: each next point must be BETTER.
        let g = vec![
            GridPoint {
                lr: 1.0,
                dropout: 0.9,
            },
            GridPoint {
                lr: 0.5,
                dropout: 0.5,
            },
            GridPoint {
                lr: 0.1,
                dropout: 0.3,
            },
            GridPoint {
                lr: 0.003,
                dropout: 0.25,
            },
        ];
        let run = grid_search_with_patience(&g, 2);
        assert!(!run.stopped_early);
    }

    #[test]
    fn deterministic_results() {
        let lrs = vec![1e-3, 3e-3];
        let drops = vec![0.1, 0.25];
        let g = cartesian_grid(&lrs, &drops);
        let r1 = grid_search_with_patience(&g, 5);
        let r2 = grid_search_with_patience(&g, 5);
        assert_eq!(r1.n_evaluated, r2.n_evaluated);
        assert_eq!(r1.stopped_early, r2.stopped_early);
    }
}
