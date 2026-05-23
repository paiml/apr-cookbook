//! # TSP — Compare Tabu Search vs Genetic Algorithm
//!
//! Same instance, two different solvers from `aprender_tsp`: TabuSolver
//! (local-search neighborhood with tabu memory) and GaSolver (population
//! evolution with crossover + mutation). Run both with matched iteration
//! budgets and compare the resulting tour lengths.
//!
//! Useful as a reference when picking a solver: TabuSolver is fast on
//! small Euclidean instances; GaSolver scales better to larger / harder
//! ones at the cost of higher per-iteration work.
//!
//! Demonstrates the **TSP.3** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley. ISBN: 978-0201157673
//!
//! Run with: cargo run --example tsp_compare_tabu_vs_genetic
//!
//! Added by PMAT-080 (expand-cookbooks: aprender-tsp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_tsp::instance::TspInstance;
use aprender_tsp::solver::{Budget, GaSolver, TabuSolver, TspSolver};

const SEED: u64 = 42;
const ITERATIONS: usize = 100;

fn coords() -> Vec<(f64, f64)> {
    // 8-city ring with two interior points — enough complexity to differentiate
    // local-search from population-evolution algorithms.
    vec![
        (0.0, 0.0),
        (2.0, 0.0),
        (3.0, 1.0),
        (3.0, 3.0),
        (2.0, 4.0),
        (0.0, 4.0),
        (1.0, 2.0),
        (2.0, 2.0),
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_compare_tabu_vs_genetic")?;

    let instance = TspInstance::from_coords("cookbook-8cities", coords())
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("instance: {e}")))?;

    let mut tabu = TabuSolver::new().with_seed(SEED);
    let tabu_solution = tabu
        .solve(&instance, Budget::Iterations(ITERATIONS))
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("tabu: {e}")))?;

    let mut ga = GaSolver::new();
    let ga_solution = ga
        .solve(&instance, Budget::Iterations(ITERATIONS))
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("ga: {e}")))?;

    println!("Solver comparison ({ITERATIONS} iters each, instance: 8-city Euclidean):");
    println!(
        "  TabuSolver: length = {:.4}  tour = {:?}",
        tabu_solution.length, tabu_solution.tour
    );
    println!(
        "  GaSolver:   length = {:.4}  tour = {:?}",
        ga_solution.length, ga_solution.tour
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn both_solvers_visit_every_city() {
        let instance = TspInstance::from_coords("test", coords()).unwrap();
        let mut tabu = TabuSolver::new().with_seed(SEED);
        let mut ga = GaSolver::new();
        for sol in [
            tabu.solve(&instance, Budget::Iterations(50)).unwrap(),
            ga.solve(&instance, Budget::Iterations(50)).unwrap(),
        ] {
            let n = coords().len();
            assert_eq!(sol.tour.len(), n);
            let mut seen = vec![false; n];
            for city in &sol.tour {
                assert!(!seen[*city]);
                seen[*city] = true;
            }
        }
    }

    #[test]
    fn both_solvers_produce_finite_lengths() {
        let instance = TspInstance::from_coords("test", coords()).unwrap();
        let mut tabu = TabuSolver::new().with_seed(SEED);
        let tabu_len = tabu
            .solve(&instance, Budget::Iterations(50))
            .unwrap()
            .length;
        let mut ga = GaSolver::new();
        let ga_len = ga.solve(&instance, Budget::Iterations(50)).unwrap().length;
        assert!(tabu_len.is_finite() && tabu_len > 0.0);
        assert!(ga_len.is_finite() && ga_len > 0.0);
    }
}
