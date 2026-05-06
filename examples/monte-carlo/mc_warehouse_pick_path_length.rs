//! # Monte-Carlo Warehouse Pick Path Length
//!
//! Sim warehouse picker traversing N random aisle positions in a
//! grid. Compares S-shape (full sweep) to Return policy (in-out).
//! Reports mean path length per policy.
//!
//! Demonstrates the **MC.95** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Roodbergen & Petersen, "Routing Order Pickers in a
//!  Warehouse" (J Operational Research 2010).
//!
//! Run with: cargo run --example mc_warehouse_pick_path_length
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WarehouseVerdict {
    Ok {
        s_shape_avg_length: f64,
        return_avg_length: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    trials: u32,
    aisles: u32,
    aisle_length: u32,
    pick_count: u32,
    seed: u64,
) -> WarehouseVerdict {
    if trials == 0 || aisles < 2 || aisle_length == 0 || pick_count == 0 {
        return WarehouseVerdict::InvalidConfig;
    }
    let mut total_s: u64 = 0;
    let mut total_r: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        // Generate pick positions: (aisle, position-in-aisle).
        let mut picks: Vec<(u32, u32)> = Vec::with_capacity(pick_count as usize);
        for _ in 0..pick_count {
            let a = ((lcg(&mut rng_state) >> 32) as u32) % aisles;
            let p = ((lcg(&mut rng_state) >> 32) as u32) % aisle_length;
            picks.push((a, p));
        }
        total_s += s_shape_path(&picks, aisles, aisle_length) as u64;
        total_r += return_path(&picks, aisle_length) as u64;
    }
    WarehouseVerdict::Ok {
        s_shape_avg_length: total_s as f64 / f64::from(trials),
        return_avg_length: total_r as f64 / f64::from(trials),
    }
}

fn s_shape_path(picks: &[(u32, u32)], aisles: u32, aisle_length: u32) -> u32 {
    // Visit aisles in order; at each aisle visit, full sweep up or down.
    let mut visited_aisles: Vec<u32> = picks.iter().map(|p| p.0).collect();
    visited_aisles.sort_unstable();
    visited_aisles.dedup();
    if visited_aisles.is_empty() {
        return 0;
    }
    // Cross-aisle movement.
    let cross = (visited_aisles.len() as u32 - 1)
        + visited_aisles[0]
        + (aisles - 1 - *visited_aisles.last().unwrap());
    // Each visited aisle: full sweep = aisle_length.
    let sweep = visited_aisles.len() as u32 * aisle_length;
    cross + sweep
}

fn return_path(picks: &[(u32, u32)], aisle_length: u32) -> u32 {
    // Visit each pick: enter aisle, go to pick depth, return to start.
    // Simplified: for each unique aisle, max pick depth × 2.
    let mut max_depth_per_aisle: std::collections::BTreeMap<u32, u32> =
        std::collections::BTreeMap::new();
    for (aisle, pos) in picks {
        let entry = max_depth_per_aisle.entry(*aisle).or_insert(0);
        if *pos > *entry {
            *entry = *pos;
        }
    }
    let mut total: u32 = 0;
    for depth in max_depth_per_aisle.values() {
        total += 2 * depth;
    }
    // Add cross-aisle traversal.
    total += max_depth_per_aisle.len() as u32 * aisle_length / 2;
    total
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_warehouse_pick_path_length")?;

    println!("typical: {:?}", simulate(100, 10, 50, 5, 42));
    println!("invalid: {:?}", simulate(0, 10, 50, 5, 42));
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
    fn paths_positive() {
        let v = simulate(20, 10, 50, 5, 42);
        if let WarehouseVerdict::Ok {
            s_shape_avg_length,
            return_avg_length,
        } = v
        {
            assert!(s_shape_avg_length > 0.0);
            assert!(return_avg_length > 0.0);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 50, 5, 42), WarehouseVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_aisles_too_few() {
        assert_eq!(simulate(10, 1, 50, 5, 42), WarehouseVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_aisle_length() {
        assert_eq!(simulate(10, 10, 0, 5, 42), WarehouseVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_picks() {
        assert_eq!(simulate(10, 10, 50, 0, 42), WarehouseVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 10, 50, 5, 42);
        let b = simulate(20, 10, 50, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn more_picks_more_path() {
        let small = simulate(20, 10, 50, 2, 42);
        let big = simulate(20, 10, 50, 20, 42);
        if let (
            WarehouseVerdict::Ok {
                s_shape_avg_length: s,
                ..
            },
            WarehouseVerdict::Ok {
                s_shape_avg_length: l,
                ..
            },
        ) = (small, big)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn longer_aisles_longer_paths() {
        let short = simulate(20, 10, 10, 5, 42);
        let long = simulate(20, 10, 100, 5, 42);
        if let (
            WarehouseVerdict::Ok {
                s_shape_avg_length: s,
                ..
            },
            WarehouseVerdict::Ok {
                s_shape_avg_length: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(20, 10, 50, 5, 42);
        if let WarehouseVerdict::Ok {
            s_shape_avg_length,
            return_avg_length,
        } = v
        {
            assert!(s_shape_avg_length.is_finite());
            assert!(return_avg_length.is_finite());
        }
    }

    #[test]
    fn single_trial_works() {
        let v = simulate(1, 10, 50, 5, 42);
        assert!(matches!(v, WarehouseVerdict::Ok { .. }));
    }
}
