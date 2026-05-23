//! # TSP — Solve with Tabu Search
//!
//! Build a small Euclidean TSP instance from inline city coordinates, run
//! `aprender_tsp::TabuSolver` for a fixed iteration budget, and report the
//! tour cost. Solver state is seeded for reproducibility (IIUR
//! determinism); same seed → same tour cost across runs.
//!
//! Demonstrates the **TSP.1** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Glover, F. (1989). Tabu Search — Part I. ORSA Journal on Computing 1(3). DOI: 10.1287/ijoc.1.3.190
//!
//! Run with: cargo run --example tsp_solve_with_tabu
//!
//! Added by PMAT-080 (expand-cookbooks: aprender-tsp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_tsp::instance::TspInstance;
use aprender_tsp::solver::{Budget, TabuSolver, TspSolver};

const SEED: u64 = 42;
const ITERATIONS: usize = 100;

/// 10-city instance arranged roughly on the unit square — small enough that
/// Tabu Search converges in <100 iterations on a single core.
fn coords() -> Vec<(f64, f64)> {
    vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 1.5),
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_solve_with_tabu")?;

    let instance = TspInstance::from_coords("cookbook-10cities", coords())
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("instance: {e}")))?;

    let mut solver = TabuSolver::new().with_seed(SEED).with_tenure(7);
    let solution = solver
        .solve(&instance, Budget::Iterations(ITERATIONS))
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("solve: {e}")))?;

    println!(
        "Tabu Search ({} iters, seed={}): tour length = {:.4}",
        ITERATIONS, SEED, solution.length
    );
    println!("  tour = {:?}", solution.tour);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn tour_visits_every_city_exactly_once() {
        let instance = TspInstance::from_coords("test", coords()).unwrap();
        let mut solver = TabuSolver::new().with_seed(SEED);
        let solution = solver
            .solve(&instance, Budget::Iterations(ITERATIONS))
            .unwrap();
        let n = coords().len();
        assert_eq!(solution.tour.len(), n);
        let mut seen = vec![false; n];
        for city in &solution.tour {
            assert!(!seen[*city], "city {city} visited twice");
            seen[*city] = true;
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let instance = TspInstance::from_coords("test", coords()).unwrap();
        let mut s1 = TabuSolver::new().with_seed(SEED);
        let mut s2 = TabuSolver::new().with_seed(SEED);
        let len1 = s1
            .solve(&instance, Budget::Iterations(ITERATIONS))
            .unwrap()
            .length;
        let len2 = s2
            .solve(&instance, Budget::Iterations(ITERATIONS))
            .unwrap()
            .length;
        assert!(
            (len1 - len2).abs() < 1e-9,
            "same seed must produce same tour length"
        );
    }
}
