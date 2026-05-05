# TSP — `aprender-tsp` Local Optimization

Recipes for [`aprender-tsp`](https://crates.io/crates/aprender-tsp) v0.31.2 — Traveling-Salesman-Problem solver that uses personalized `.apr` models to bias edge selection from user history (delivery routes, daily commutes, etc.). All-local, no network.

Closes the **≥3 recipes per sister crate** requirement from [expand-cookbooks/subcrate-coverage.md](../../specifications/expand-cookbooks/subcrate-coverage.md).

## Recipes

| # | Recipe | What |
|---|--------|------|
| TSP.1 | [`tsp_solve_with_tabu`](https://github.com/paiml/apr-cookbook/blob/main/examples/tsp/tsp_solve_with_tabu.rs) | 10-city Euclidean TSP solved by `TabuSolver` with seed-deterministic output |
| TSP.2 | [`tsp_distance_matrix_explicit`](https://github.com/paiml/apr-cookbook/blob/main/examples/tsp/tsp_distance_matrix_explicit.rs) | 5-city symmetric distance matrix (non-Euclidean) via `TspInstance::from_matrix` + jagged/empty rejection |
| TSP.3 | [`tsp_compare_tabu_vs_genetic`](https://github.com/paiml/apr-cookbook/blob/main/examples/tsp/tsp_compare_tabu_vs_genetic.rs) | Same instance solved by `TabuSolver` vs `GaSolver` with matched iteration budget |

## API surface exercised

- `aprender_tsp::instance::TspInstance::{from_coords, from_matrix}`
- `aprender_tsp::solver::{TabuSolver, GaSolver, TspSolver, Budget}`
- `TabuSolver::with_seed(u64)` — deterministic output for IIUR

## Provenance

Added during PMAT-080 (expand-cookbooks initiative, v6.1.0).
