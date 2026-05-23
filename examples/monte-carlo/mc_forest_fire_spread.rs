//! # Monte-Carlo Forest Fire Spread
//!
//! Sim 2D forest-fire model: trees burn neighbors with probability p.
//! Tracks how many trees burn before fire dies out. Returns burned
//! count and spread radius.
//!
//! Demonstrates the **MC.191** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Drossel & Schwabl, "Self-organized critical forest-fire
//!  model" Phys. Rev. Lett. 69(11) (1992); BTW critical phenomenon.
//!
//! Run with: cargo run --example mc_forest_fire_spread
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum FireVerdict {
    Ok {
        burned_count: u32,
        spread_radius: u32,
    },
    InvalidConfig,
}

pub fn simulate(grid_size: u32, spread_prob_pct: u32, seed: u64) -> FireVerdict {
    if !(5..=100).contains(&grid_size) || !(1..=99).contains(&spread_prob_pct) {
        return FireVerdict::InvalidConfig;
    }
    let n = grid_size as usize;
    let p = spread_prob_pct as f64 / 100.0;
    let mut state = seed | 1;
    let mut burned = vec![false; n * n];
    let center = (n / 2) * n + n / 2;
    let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
    queue.push_back((center, 0));
    burned[center] = true;
    let mut burned_count = 1u32;
    let mut max_radius = 0u32;
    while let Some((cell, dist)) = queue.pop_front() {
        if dist > max_radius {
            max_radius = dist;
        }
        let row = cell / n;
        let col = cell % n;
        let neighbors: Vec<usize> = [
            (row.wrapping_sub(1), col),
            (row + 1, col),
            (row, col.wrapping_sub(1)),
            (row, col + 1),
        ]
        .iter()
        .filter_map(|(r, c)| {
            if *r < n && *c < n {
                Some(*r * n + *c)
            } else {
                None
            }
        })
        .collect();
        for nb in neighbors {
            if burned[nb] {
                continue;
            }
            let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
            if r < p {
                burned[nb] = true;
                burned_count += 1;
                queue.push_back((nb, dist + 1));
            }
        }
    }
    FireVerdict::Ok {
        burned_count,
        spread_radius: max_radius,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_forest_fire_spread")?;

    println!("p=0.7: {:?}", simulate(30, 70, 42));
    println!("p=0.3: {:?}", simulate(30, 30, 42));
    println!("invalid: {:?}", simulate(2, 50, 42));
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
    fn invalid_too_small_grid() {
        assert_eq!(simulate(2, 50, 42), FireVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_prob() {
        assert_eq!(simulate(10, 0, 42), FireVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_prob() {
        assert_eq!(simulate(10, 100, 42), FireVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 50, 42);
        let b = simulate(20, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_prob_more_burned() {
        let low = simulate(30, 30, 42);
        let high = simulate(30, 70, 42);
        if let (
            FireVerdict::Ok {
                burned_count: l, ..
            },
            FireVerdict::Ok {
                burned_count: h, ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn burned_at_least_one() {
        let v = simulate(20, 50, 42);
        if let FireVerdict::Ok { burned_count, .. } = v {
            assert!(burned_count >= 1);
        }
    }

    #[test]
    fn burned_le_grid_squared() {
        let v = simulate(20, 90, 42);
        if let FireVerdict::Ok { burned_count, .. } = v {
            assert!(burned_count <= 400);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 1, 42);
        assert!(matches!(v, FireVerdict::Ok { .. }));
    }

    #[test]
    fn many_cells_handled() {
        let v = simulate(50, 50, 42);
        assert!(matches!(v, FireVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(20, 50, 42);
        let b = simulate(20, 50, 999);
        assert!(a != b);
    }

    #[test]
    fn radius_le_grid_size() {
        let v = simulate(20, 80, 42);
        if let FireVerdict::Ok { spread_radius, .. } = v {
            // From center, max radius is grid/2 + small overshoot.
            assert!(spread_radius <= 40);
        }
    }
}
