//! # TSP — Solve from Explicit Distance Matrix
//!
//! Build a TSP instance from an explicit symmetric distance matrix (no
//! coordinates), useful when distances aren't Euclidean (e.g. road-network
//! routing, time-of-travel matrices, asymmetric flight times). The
//! `aprender_tsp::TspInstance::from_matrix` API takes a `Vec<Vec<f64>>`
//! and rejects malformed input via `TspError::InvalidInstance`.
//!
//! Demonstrates the **TSP.2** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Reinelt, G. (1991). TSPLIB — A Traveling Salesman Problem Library. ORSA Journal on Computing 3(4). DOI: 10.1287/ijoc.3.4.376
//!
//! Run with: cargo run --example tsp_distance_matrix_explicit
//!
//! Added by PMAT-080 (expand-cookbooks: aprender-tsp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_tsp::instance::TspInstance;
use aprender_tsp::solver::{Budget, TabuSolver, TspSolver};

const SEED: u64 = 42;

/// Symmetric 5-city distance matrix. Off-diagonal entries are travel costs
/// (e.g. minutes by car); diagonal is zero. Values picked so a unique optimal
/// tour exists at length 13: 0 → 2 → 4 → 3 → 1 → 0 (= 2+1+3+5+2 = 13).
fn distances() -> Vec<Vec<f64>> {
    vec![
        //      0    1    2    3    4
        vec![0.0, 2.0, 2.0, 5.0, 4.0], // 0
        vec![2.0, 0.0, 3.0, 5.0, 4.0], // 1
        vec![2.0, 3.0, 0.0, 4.0, 1.0], // 2
        vec![5.0, 5.0, 4.0, 0.0, 3.0], // 3
        vec![4.0, 4.0, 1.0, 3.0, 0.0], // 4
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_distance_matrix_explicit")?;

    let instance = TspInstance::from_matrix("cookbook-5cities-matrix", distances())
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("from_matrix: {e}")))?;

    let mut solver = TabuSolver::new().with_seed(SEED);
    let solution = solver
        .solve(&instance, Budget::Iterations(50))
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("solve: {e}")))?;

    println!(
        "5-city matrix TSP (TabuSolver, 50 iters, seed={}): tour length = {:.2}",
        SEED, solution.length
    );
    println!("  tour = {:?}", solution.tour);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_solve_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn jagged_matrix_is_rejected() {
        // Row length must equal dimension; jagged rows = invalid.
        let bad = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]];
        let err = TspInstance::from_matrix("bad", bad);
        assert!(err.is_err(), "jagged distance matrix must be rejected");
    }

    #[test]
    fn empty_matrix_is_rejected() {
        let err = TspInstance::from_matrix("empty", Vec::<Vec<f64>>::new());
        assert!(err.is_err(), "empty distance matrix must be rejected");
    }
}
