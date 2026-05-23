//! # Monte-Carlo Geo Distance Routing
//!
//! Sim N random points in 2D unit square; route each to closest of M
//! data-center centers (uniformly placed). Reports avg routing
//! distance and max.
//!
//! Demonstrates the **MC.104** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: nearest-center clustering (k-means assignment step);
//!  CDN PoP routing convention.
//!
//! Run with: cargo run --example mc_geo_distance_routing
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GeoVerdict {
    Ok {
        avg_distance: f64,
        max_distance: f64,
    },
    InvalidConfig,
}

pub fn simulate(points: u32, centers: u32, seed: u64) -> GeoVerdict {
    if points == 0 || centers == 0 {
        return GeoVerdict::InvalidConfig;
    }
    let center_positions: Vec<(f64, f64)> = (0..centers)
        .map(|i| {
            let cell = f64::from(i + 1) / f64::from(centers + 1);
            (cell, 0.5)
        })
        .collect();
    let mut total_distance: f64 = 0.0;
    let mut max_distance: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..points {
        let x = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let y = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let mut min_d = f64::INFINITY;
        for &(cx, cy) in &center_positions {
            let dx = x - cx;
            let dy = y - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < min_d {
                min_d = d;
            }
        }
        total_distance += min_d;
        if min_d > max_distance {
            max_distance = min_d;
        }
    }
    GeoVerdict::Ok {
        avg_distance: total_distance / f64::from(points),
        max_distance,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_geo_distance_routing")?;

    println!("3 centers: {:?}", simulate(1000, 3, 42));
    println!("10 centers: {:?}", simulate(1000, 10, 42));
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
    fn more_centers_lower_avg_distance() {
        let few = simulate(2000, 3, 42);
        let many = simulate(2000, 30, 42);
        if let (
            GeoVerdict::Ok {
                avg_distance: f, ..
            },
            GeoVerdict::Ok {
                avg_distance: m, ..
            },
        ) = (few, many)
        {
            assert!(m < f);
        }
    }

    #[test]
    fn invalid_zero_points() {
        assert_eq!(simulate(0, 3, 42), GeoVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_centers() {
        assert_eq!(simulate(100, 0, 42), GeoVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 42);
        let b = simulate(500, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn distances_finite() {
        let v = simulate(500, 5, 42);
        if let GeoVerdict::Ok {
            avg_distance,
            max_distance,
        } = v
        {
            assert!(avg_distance.is_finite());
            assert!(max_distance.is_finite());
        }
    }

    #[test]
    fn avg_le_max() {
        let v = simulate(500, 5, 42);
        if let GeoVerdict::Ok {
            avg_distance,
            max_distance,
        } = v
        {
            assert!(avg_distance <= max_distance);
        }
    }

    #[test]
    fn distances_in_realistic_bounds() {
        // Unit square: max possible distance = sqrt(2) ≈ 1.41.
        let v = simulate(500, 5, 42);
        if let GeoVerdict::Ok { max_distance, .. } = v {
            assert!(max_distance < 1.5);
        }
    }

    #[test]
    fn distances_nonneg() {
        let v = simulate(500, 5, 42);
        if let GeoVerdict::Ok {
            avg_distance,
            max_distance,
        } = v
        {
            assert!(avg_distance >= 0.0);
            assert!(max_distance >= 0.0);
        }
    }

    #[test]
    fn single_center_higher_distance() {
        let one = simulate(1000, 1, 42);
        let three = simulate(1000, 3, 42);
        if let (
            GeoVerdict::Ok {
                avg_distance: o, ..
            },
            GeoVerdict::Ok {
                avg_distance: t, ..
            },
        ) = (one, three)
        {
            assert!(o > t);
        }
    }

    #[test]
    fn single_point_works() {
        let v = simulate(1, 5, 42);
        if let GeoVerdict::Ok {
            avg_distance,
            max_distance,
        } = v
        {
            assert!((avg_distance - max_distance).abs() < 1e-9);
        }
    }

    #[test]
    fn many_points_handled() {
        let v = simulate(10_000, 10, 42);
        assert!(matches!(v, GeoVerdict::Ok { .. }));
    }
}
