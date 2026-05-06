//! # TSP Nearest-Neighbor Heuristic
//!
//! Greedy construction: start at any city, repeatedly visit nearest
//! unvisited neighbor, return to start. Worst-case O(log n) factor of
//! optimal in random metric instances. This recipe builds the
//! construction over an explicit distance matrix.
//!
//! Demonstrates the **TSP.5** recipe for PMAT-129 (tsp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Rosenkrantz et al. (1977). Approximate algorithms for the TSP.
//!
//! Run with: cargo run --example tsp_nearest_neighbor
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NnVerdict {
    Ok { tour: Vec<usize>, length: f64 },
    EmptyMatrix,
    NonSquareMatrix,
    StartOutOfRange,
    InvalidDistance,
}

pub fn nearest_neighbor(matrix: &[Vec<f64>], start: usize) -> NnVerdict {
    let n = matrix.len();
    if n == 0 {
        return NnVerdict::EmptyMatrix;
    }
    if matrix.iter().any(|row| row.len() != n) {
        return NnVerdict::NonSquareMatrix;
    }
    if start >= n {
        return NnVerdict::StartOutOfRange;
    }
    if matrix.iter().flatten().any(|d| !d.is_finite() || *d < 0.0) {
        return NnVerdict::InvalidDistance;
    }
    let mut visited = vec![false; n];
    let mut tour = Vec::with_capacity(n + 1);
    let mut current = start;
    visited[current] = true;
    tour.push(current);
    let mut total = 0.0;
    for _ in 1..n {
        let mut best = (usize::MAX, f64::INFINITY);
        for (j, &d) in matrix[current].iter().enumerate() {
            if !visited[j] && d < best.1 {
                best = (j, d);
            }
        }
        if best.0 == usize::MAX {
            return NnVerdict::InvalidDistance;
        }
        visited[best.0] = true;
        tour.push(best.0);
        total += best.1;
        current = best.0;
    }
    // Return to start.
    total += matrix[current][start];
    tour.push(start);
    NnVerdict::Ok {
        tour,
        length: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_nearest_neighbor")?;

    let m = vec![
        vec![0.0, 10.0, 15.0, 20.0],
        vec![10.0, 0.0, 35.0, 25.0],
        vec![15.0, 35.0, 0.0, 30.0],
        vec![20.0, 25.0, 30.0, 0.0],
    ];
    println!("4-city: {:?}", nearest_neighbor(&m, 0));
    println!("empty: {:?}", nearest_neighbor(&[], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 10.0, 15.0, 20.0],
            vec![10.0, 0.0, 35.0, 25.0],
            vec![15.0, 35.0, 0.0, 30.0],
            vec![20.0, 25.0, 30.0, 0.0],
        ]
    }

    #[test]
    fn nn_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_4_city_returns_full_tour() {
        if let NnVerdict::Ok { tour, .. } = nearest_neighbor(&sample(), 0) {
            // Tour visits all 4 cities + returns to start = 5 entries.
            assert_eq!(tour.len(), 5);
            assert_eq!(tour[0], 0);
            assert_eq!(tour[4], 0);
        }
    }

    #[test]
    fn tour_visits_all_cities_exactly_once_before_return() {
        if let NnVerdict::Ok { tour, .. } = nearest_neighbor(&sample(), 0) {
            let mut without_return: Vec<usize> = tour[..tour.len() - 1].to_vec();
            without_return.sort_unstable();
            assert_eq!(without_return, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn nearest_first_chosen() {
        // From city 0: nearest is city 1 (distance 10).
        if let NnVerdict::Ok { tour, .. } = nearest_neighbor(&sample(), 0) {
            assert_eq!(tour[1], 1);
        }
    }

    #[test]
    fn empty_matrix_rejected() {
        let m: Vec<Vec<f64>> = vec![];
        assert_eq!(nearest_neighbor(&m, 0), NnVerdict::EmptyMatrix);
    }

    #[test]
    fn non_square_matrix_rejected() {
        let m = vec![vec![0.0, 1.0], vec![1.0]];
        assert_eq!(nearest_neighbor(&m, 0), NnVerdict::NonSquareMatrix);
    }

    #[test]
    fn start_out_of_range_rejected() {
        assert_eq!(nearest_neighbor(&sample(), 10), NnVerdict::StartOutOfRange);
    }

    #[test]
    fn negative_distance_rejected() {
        let mut m = sample();
        m[0][1] = -1.0;
        assert_eq!(nearest_neighbor(&m, 0), NnVerdict::InvalidDistance);
    }

    #[test]
    fn nan_distance_rejected() {
        let mut m = sample();
        m[1][2] = f64::NAN;
        assert_eq!(nearest_neighbor(&m, 0), NnVerdict::InvalidDistance);
    }

    #[test]
    fn single_city_returns_trivial_tour() {
        let m = vec![vec![0.0]];
        if let NnVerdict::Ok { tour, length } = nearest_neighbor(&m, 0) {
            assert_eq!(tour, vec![0, 0]);
            assert!((length - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn different_starts_may_yield_different_tours() {
        // Just verify no panic and both produce valid tours.
        let from_0 = nearest_neighbor(&sample(), 0);
        let from_2 = nearest_neighbor(&sample(), 2);
        assert!(matches!(from_0, NnVerdict::Ok { .. }));
        assert!(matches!(from_2, NnVerdict::Ok { .. }));
    }
}
