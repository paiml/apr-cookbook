//! # Monte-Carlo TSP Random-Search Tour
//!
//! Sample N random tour permutations of n cities (closed loop) and
//! return the shortest tour length found. Cities are 2D points with
//! Euclidean distance.
//!
//! Demonstrates the **MC.135** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lawler et al., The Traveling Salesman Problem (1985);
//!  Reinelt, TSPLIB95 benchmark.
//!
//! Run with: cargo run --example mc_tsp_random_search
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TspVerdict {
    Ok {
        best_length: u32,
        samples_evaluated: u32,
    },
    InvalidConfig,
}

pub fn simulate(cities: &[(i32, i32)], samples: u32, seed: u64) -> TspVerdict {
    if cities.len() < 3 || samples == 0 {
        return TspVerdict::InvalidConfig;
    }
    let n = cities.len();
    let mut state = seed | 1;
    let mut best_length = u32::MAX;
    for _ in 0..samples {
        // Fisher-Yates shuffle on indices 0..n
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = ((lcg(&mut state) >> 32) as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut total = 0u32;
        for k in 0..n {
            let a = cities[perm[k]];
            let b = cities[perm[(k + 1) % n]];
            total = total.saturating_add(dist(a, b));
        }
        if total < best_length {
            best_length = total;
        }
    }
    TspVerdict::Ok {
        best_length,
        samples_evaluated: samples,
    }
}

fn dist(a: (i32, i32), b: (i32, i32)) -> u32 {
    let dx = (a.0 - b.0) as f64;
    let dy = (a.1 - b.1) as f64;
    (dx * dx + dy * dy).sqrt() as u32
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_tsp_random_search")?;

    let cities = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)];
    println!("tour-search: {:?}", simulate(&cities, 1000, 42));
    println!("invalid: {:?}", simulate(&[(0, 0)], 100, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_too_few_cities() {
        assert_eq!(
            simulate(&[(0, 0), (1, 1)], 100, 42),
            TspVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(&[(0, 0), (1, 1), (2, 2)], 0, 42),
            TspVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let cities = [(0, 0), (1, 1), (2, 2)];
        let a = simulate(&cities, 50, 42);
        let b = simulate(&cities, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn samples_evaluated_returned() {
        let cities = [(0, 0), (1, 1), (2, 2)];
        let v = simulate(&cities, 50, 42);
        if let TspVerdict::Ok {
            samples_evaluated, ..
        } = v
        {
            assert_eq!(samples_evaluated, 50);
        }
    }

    #[test]
    fn best_length_finite() {
        let cities = [(0, 0), (10, 0), (10, 10), (0, 10)];
        let v = simulate(&cities, 1000, 42);
        if let TspVerdict::Ok { best_length, .. } = v {
            assert!(best_length < u32::MAX);
        }
    }

    #[test]
    fn square_optimum_near_40() {
        // Unit square corners: optimal TSP = perimeter ≈ 40.
        let cities = [(0, 0), (10, 0), (10, 10), (0, 10)];
        let v = simulate(&cities, 5000, 42);
        if let TspVerdict::Ok { best_length, .. } = v {
            assert_eq!(best_length, 40);
        }
    }

    #[test]
    fn collinear_cities_handled() {
        let cities = [(0, 0), (1, 0), (2, 0), (3, 0)];
        let v = simulate(&cities, 100, 42);
        assert!(matches!(v, TspVerdict::Ok { .. }));
    }

    #[test]
    fn more_samples_better_or_equal() {
        let cities = [(0, 0), (10, 0), (5, 8), (3, 3)];
        let small = simulate(&cities, 10, 42);
        let large = simulate(&cities, 1000, 42);
        if let (TspVerdict::Ok { best_length: s, .. }, TspVerdict::Ok { best_length: l, .. }) =
            (small, large)
        {
            assert!(l <= s);
        }
    }

    #[test]
    fn single_seed_consistency() {
        let cities = [(0, 0), (1, 1), (2, 2)];
        let v1 = simulate(&cities, 100, 999);
        let v2 = simulate(&cities, 100, 999);
        assert_eq!(v1, v2);
    }

    #[test]
    fn many_cities_handled() {
        let cities: Vec<(i32, i32)> = (0..15).map(|i| (i, i * 2)).collect();
        let v = simulate(&cities, 100, 42);
        assert!(matches!(v, TspVerdict::Ok { .. }));
    }

    #[test]
    fn three_cities_handled() {
        let cities = [(0, 0), (3, 0), (0, 4)];
        let v = simulate(&cities, 100, 42);
        if let TspVerdict::Ok { best_length, .. } = v {
            // Triangle perimeter = 3+4+5 = 12.
            assert_eq!(best_length, 12);
        }
    }
}
