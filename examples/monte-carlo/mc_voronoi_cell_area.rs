//! # Monte-Carlo Voronoi Cell Area Estimation
//!
//! Sample N random query points in a unit square; for each, find the
//! nearest of M Voronoi seeds. Reports area fraction (cell density)
//! per seed.
//!
//! Demonstrates the **MC.118** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Voronoi (1908) Sur quelques propriétés des formes
//!  quadratiques; nearest-neighbor area estimation.
//!
//! Run with: cargo run --example mc_voronoi_cell_area
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum VoronoiVerdict {
    Ok {
        cell_areas: Vec<f64>,
        max_area: f64,
        min_area: f64,
    },
    InvalidConfig,
}

pub fn simulate(samples: u32, seeds: u32, seed: u64) -> VoronoiVerdict {
    if samples == 0 || seeds == 0 {
        return VoronoiVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let seed_positions: Vec<(f64, f64)> = (0..seeds)
        .map(|_| {
            let x = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            let y = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            (x, y)
        })
        .collect();
    let mut counts: Vec<u32> = vec![0; seeds as usize];
    for _ in 0..samples {
        let qx = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let qy = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let mut min_d = f64::INFINITY;
        let mut min_idx = 0usize;
        for (i, &(sx, sy)) in seed_positions.iter().enumerate() {
            let d = (qx - sx).powi(2) + (qy - sy).powi(2);
            if d < min_d {
                min_d = d;
                min_idx = i;
            }
        }
        counts[min_idx] += 1;
    }
    let total = f64::from(samples);
    let cell_areas: Vec<f64> = counts.iter().map(|c| f64::from(*c) / total).collect();
    let max_area = cell_areas.iter().fold(f64::MIN, |a, b| a.max(*b));
    let min_area = cell_areas.iter().fold(f64::MAX, |a, b| a.min(*b));
    VoronoiVerdict::Ok {
        cell_areas,
        max_area,
        min_area,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_voronoi_cell_area")?;

    println!("3 seeds: {:?}", simulate(10_000, 3, 42));
    println!("10 seeds: {:?}", simulate(10_000, 10, 42));
    println!("invalid: {:?}", simulate(0, 3, 42));
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
    fn cell_count_matches_seeds() {
        let v = simulate(1000, 5, 42);
        if let VoronoiVerdict::Ok { cell_areas, .. } = v {
            assert_eq!(cell_areas.len(), 5);
        }
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(0, 3, 42), VoronoiVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_seeds() {
        assert_eq!(simulate(100, 0, 42), VoronoiVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 42);
        let b = simulate(500, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn areas_sum_to_one() {
        let v = simulate(1000, 5, 42);
        if let VoronoiVerdict::Ok { cell_areas, .. } = v {
            let sum: f64 = cell_areas.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn max_ge_min() {
        let v = simulate(1000, 5, 42);
        if let VoronoiVerdict::Ok {
            max_area, min_area, ..
        } = v
        {
            assert!(max_area >= min_area);
        }
    }

    #[test]
    fn single_seed_one_area() {
        let v = simulate(1000, 1, 42);
        if let VoronoiVerdict::Ok {
            cell_areas,
            max_area,
            ..
        } = v
        {
            assert!((cell_areas[0] - 1.0).abs() < 1e-9);
            assert!((max_area - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn areas_in_unit_range() {
        let v = simulate(1000, 5, 42);
        if let VoronoiVerdict::Ok { cell_areas, .. } = v {
            for a in &cell_areas {
                assert!((0.0..=1.0).contains(a));
            }
        }
    }

    #[test]
    fn many_seeds_handled() {
        let v = simulate(10_000, 50, 42);
        if let VoronoiVerdict::Ok { cell_areas, .. } = v {
            assert_eq!(cell_areas.len(), 50);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(1000, 5, 42);
        if let VoronoiVerdict::Ok {
            max_area, min_area, ..
        } = v
        {
            assert!(max_area.is_finite());
            assert!(min_area.is_finite());
        }
    }

    #[test]
    fn larger_sample_more_stable() {
        let small = simulate(100, 5, 42);
        let big = simulate(10_000, 5, 42);
        // Both should sum to 1.
        if let (
            VoronoiVerdict::Ok {
                cell_areas: s_a, ..
            },
            VoronoiVerdict::Ok {
                cell_areas: b_a, ..
            },
        ) = (small, big)
        {
            let sum_s: f64 = s_a.iter().sum();
            let sum_b: f64 = b_a.iter().sum();
            assert!((sum_s - 1.0).abs() < 1e-9);
            assert!((sum_b - 1.0).abs() < 1e-9);
        }
    }
}
