//! # TSP 2-opt Local Improvement
//!
//! 2-opt: reverse a segment of the tour, accept the change if it
//! shortens the total length. Repeat until no improvement for a full
//! pass. This recipe builds the single-swap improvement check + the
//! tour-delta calculator.
//!
//! Demonstrates the **TSP.6** recipe for PMAT-129 (tsp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Croes, G. A. (1958). A method for solving traveling-salesman problems.
//!
//! Run with: cargo run --example tsp_two_opt_swap_improver
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SwapVerdict {
    Improved { delta: f64, new_tour: Vec<usize> },
    NoImprovement,
    InvalidIndex,
    EmptyTour,
}

/// Compute the change in tour length if edges (i,i+1) and (j,j+1) are
/// reversed (i.e., 2-opt swap). Returns `new - old`; negative means
/// improvement.
pub fn swap_delta(tour: &[usize], matrix: &[Vec<f64>], i: usize, j: usize) -> Option<f64> {
    if i + 1 >= j || j >= tour.len() - 1 {
        return None;
    }
    let a = tour[i];
    let b = tour[i + 1];
    let c = tour[j];
    let d = tour[j + 1];
    if a >= matrix.len() || b >= matrix.len() || c >= matrix.len() || d >= matrix.len() {
        return None;
    }
    let old = matrix[a][b] + matrix[c][d];
    let new = matrix[a][c] + matrix[b][d];
    Some(new - old)
}

pub fn try_swap(tour: &[usize], matrix: &[Vec<f64>], i: usize, j: usize) -> SwapVerdict {
    if tour.is_empty() {
        return SwapVerdict::EmptyTour;
    }
    let Some(delta) = swap_delta(tour, matrix, i, j) else {
        return SwapVerdict::InvalidIndex;
    };
    if delta < -1e-9 {
        let mut new_tour = tour.to_vec();
        new_tour[i + 1..=j].reverse();
        SwapVerdict::Improved { delta, new_tour }
    } else {
        SwapVerdict::NoImprovement
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_two_opt_swap_improver")?;

    let m = vec![
        vec![0.0, 10.0, 100.0, 20.0],
        vec![10.0, 0.0, 35.0, 100.0],
        vec![100.0, 35.0, 0.0, 30.0],
        vec![20.0, 100.0, 30.0, 0.0],
    ];
    let bad_tour = vec![0, 2, 1, 3, 0];
    println!("delta: {:?}", swap_delta(&bad_tour, &m, 0, 2));
    println!("try_swap: {:?}", try_swap(&bad_tour, &m, 0, 2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 10.0, 100.0, 20.0],
            vec![10.0, 0.0, 35.0, 100.0],
            vec![100.0, 35.0, 0.0, 30.0],
            vec![20.0, 100.0, 30.0, 0.0],
        ]
    }

    #[test]
    fn improver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn swap_with_negative_delta_improves() {
        // Tour 0-2-1-3-0: bad ordering. Swap (i=0, j=2) reverses [2,1] → 0-1-2-3-0.
        let bad = vec![0usize, 2, 1, 3, 0];
        let v = try_swap(&bad, &sample_matrix(), 0, 2);
        if let SwapVerdict::Improved { delta, new_tour } = v {
            assert!(delta < 0.0);
            assert_eq!(new_tour, vec![0, 1, 2, 3, 0]);
        } else {
            panic!("expected Improved");
        }
    }

    #[test]
    fn already_optimal_no_improvement() {
        // Tour 0-1-2-3-0 is already good; swap (i=0, j=2) would worsen it.
        let good = vec![0usize, 1, 2, 3, 0];
        let v = try_swap(&good, &sample_matrix(), 0, 2);
        assert_eq!(v, SwapVerdict::NoImprovement);
    }

    #[test]
    fn empty_tour_rejected() {
        let m = vec![vec![0.0]];
        assert_eq!(try_swap(&[], &m, 0, 1), SwapVerdict::EmptyTour);
    }

    #[test]
    fn out_of_bound_indices_rejected() {
        let tour = vec![0usize, 1, 2, 3, 0];
        // i+1 >= j → invalid 2-opt.
        assert_eq!(
            try_swap(&tour, &sample_matrix(), 2, 2),
            SwapVerdict::InvalidIndex
        );
        // j out of range.
        assert_eq!(
            try_swap(&tour, &sample_matrix(), 0, 10),
            SwapVerdict::InvalidIndex
        );
    }

    #[test]
    fn delta_calculation_accurate() {
        // For tour 0-2-1-3-0: edges (0,2)=100 + (1,3)=100 → swap to (0,1)=10 + (2,3)=30.
        // delta = (10+30) - (100+100) = -160.
        let bad = vec![0usize, 2, 1, 3, 0];
        let d = swap_delta(&bad, &sample_matrix(), 0, 2).unwrap();
        assert!((d - (-160.0)).abs() < 1e-9);
    }

    #[test]
    fn swap_delta_invalid_pair_returns_none() {
        let tour = vec![0usize, 1, 2, 3, 0];
        assert!(swap_delta(&tour, &sample_matrix(), 0, 0).is_none());
    }

    #[test]
    fn city_index_out_of_matrix_rejected() {
        let m = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let tour = vec![0usize, 1, 5, 0]; // 5 is out of range
        assert!(swap_delta(&tour, &m, 0, 2).is_none());
    }

    #[test]
    fn near_zero_delta_treated_as_no_improvement() {
        // Symmetric matrix where swap is exactly neutral.
        let m = vec![
            vec![0.0, 1.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 0.0],
        ];
        let tour = vec![0usize, 1, 2, 3, 0];
        // All edges = 1; any swap has delta = 0 → NoImprovement.
        assert_eq!(try_swap(&tour, &m, 0, 2), SwapVerdict::NoImprovement);
    }

    #[test]
    fn improvement_preserves_tour_length() {
        let bad = vec![0usize, 2, 1, 3, 0];
        if let SwapVerdict::Improved { new_tour, .. } = try_swap(&bad, &sample_matrix(), 0, 2) {
            assert_eq!(new_tour.len(), bad.len());
            assert_eq!(new_tour.first(), bad.first());
            assert_eq!(new_tour.last(), bad.last());
        }
    }
}
