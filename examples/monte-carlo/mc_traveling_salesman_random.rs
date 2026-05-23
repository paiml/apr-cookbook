//! # Monte-Carlo TSP Random Instance with Nearest-Neighbor
//!
//! Generate random TSP instance (N cities in unit square) and solve
//! using nearest-neighbor heuristic. Reports tour length / sqrt(N)
//! lower-bound ratio (Beardwood-Halton-Hammersley benchmark).
//!
//! Demonstrates the **MC.82** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Beardwood, Halton, Hammersley (1959) Proc. Cambridge Phil.
//!  Soc. 55; nearest-neighbor TSP heuristic (Rosenkrantz et al. 1977).
//!
//! Run with: cargo run --example mc_traveling_salesman_random
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TspVerdict {
    Ok {
        avg_tour_length: f64,
        tour_to_sqrt_n_ratio: f64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, cities: u32, seed: u64) -> TspVerdict {
    if trials == 0 || cities < 2 {
        return TspVerdict::InvalidConfig;
    }
    let mut total_length: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut points: Vec<(f64, f64)> = Vec::with_capacity(cities as usize);
        for _ in 0..cities {
            let x = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            let y = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            points.push((x, y));
        }
        total_length += nearest_neighbor_tour(&points);
    }
    let avg = total_length / f64::from(trials);
    let bhh = (cities as f64).sqrt();
    TspVerdict::Ok {
        avg_tour_length: avg,
        tour_to_sqrt_n_ratio: avg / bhh,
    }
}

fn nearest_neighbor_tour(points: &[(f64, f64)]) -> f64 {
    let n = points.len();
    let mut visited = vec![false; n];
    visited[0] = true;
    let mut current = 0usize;
    let mut total = 0.0;
    for _ in 0..(n - 1) {
        let mut best_dist = f64::INFINITY;
        let mut best_idx = 0;
        for j in 0..n {
            if !visited[j] {
                let d = euclid(points[current], points[j]);
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
        }
        total += best_dist;
        visited[best_idx] = true;
        current = best_idx;
    }
    // Close the tour.
    total += euclid(points[current], points[0]);
    total
}

fn euclid(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_traveling_salesman_random")?;

    println!("10 cities: {:?}", simulate(50, 10, 42));
    println!("50 cities: {:?}", simulate(20, 50, 42));
    println!("invalid: {:?}", simulate(0, 10, 42));
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
    fn average_tour_length_positive() {
        let v = simulate(20, 10, 42);
        if let TspVerdict::Ok {
            avg_tour_length, ..
        } = v
        {
            assert!(avg_tour_length > 0.0);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 42), TspVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_one_city() {
        assert_eq!(simulate(10, 1, 42), TspVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 10, 42);
        let b = simulate(20, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn larger_n_longer_tour() {
        let small = simulate(20, 5, 42);
        let large = simulate(20, 50, 42);
        if let (
            TspVerdict::Ok {
                avg_tour_length: s, ..
            },
            TspVerdict::Ok {
                avg_tour_length: l, ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn ratio_in_realistic_band() {
        // BHH constant ≈ 0.7124; nearest-neighbor is ~25% worse → ~0.9.
        // We allow generous bounds for finite-N.
        let v = simulate(50, 50, 42);
        if let TspVerdict::Ok {
            tour_to_sqrt_n_ratio,
            ..
        } = v
        {
            assert!(tour_to_sqrt_n_ratio > 0.5);
            assert!(tour_to_sqrt_n_ratio < 2.0);
        }
    }

    #[test]
    fn two_cities_round_trip() {
        // 2 cities: tour = 2 × dist.
        let v = simulate(20, 2, 42);
        if let TspVerdict::Ok {
            avg_tour_length, ..
        } = v
        {
            // Both points uniform on unit square: avg dist ~0.52, tour ~1.04.
            assert!(avg_tour_length > 0.0);
            assert!(avg_tour_length < 3.0);
        }
    }

    #[test]
    fn ratio_positive() {
        let v = simulate(10, 20, 42);
        if let TspVerdict::Ok {
            tour_to_sqrt_n_ratio,
            ..
        } = v
        {
            assert!(tour_to_sqrt_n_ratio > 0.0);
        }
    }

    #[test]
    fn single_trial_works() {
        let v = simulate(1, 10, 42);
        assert!(matches!(v, TspVerdict::Ok { .. }));
    }

    #[test]
    fn larger_trial_count_smooths_avg() {
        // More trials should give similar avg as fewer (within tolerance).
        let small = simulate(5, 20, 42);
        let large = simulate(100, 20, 42);
        if let (
            TspVerdict::Ok {
                avg_tour_length: s, ..
            },
            TspVerdict::Ok {
                avg_tour_length: l, ..
            },
        ) = (small, large)
        {
            assert!((s - l).abs() < 1.5);
        }
    }
}
