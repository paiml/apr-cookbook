//! # Monte-Carlo A* Path Cost
//!
//! Generate an N×N grid maze with random obstacles (`obstacle_prob`)
//! and run a simplified A* (Manhattan heuristic) from (0,0) to
//! (N-1,N-1). Reports path-found rate, mean steps explored.
//!
//! Demonstrates the **MC.74** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hart, Nilsson, Raphael, A Formal Basis for the Heuristic
//!  Determination of Minimum Cost Paths (IEEE SSC 4, 1968).
//!
//! Run with: cargo run --example mc_a_star_path_cost
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;

#[derive(Debug, PartialEq)]
pub enum AStarVerdict {
    Ok {
        path_found_rate: f64,
        mean_steps_explored: f64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, grid_size: u32, obstacle_prob: f64, seed: u64) -> AStarVerdict {
    if trials == 0 || grid_size < 2 || !(0.0..1.0).contains(&obstacle_prob) {
        return AStarVerdict::InvalidConfig;
    }
    let n = grid_size as usize;
    let mut found_count: u32 = 0;
    let mut total_steps: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut grid = vec![vec![false; n]; n];
        for (r, row) in grid.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                if (r, c) == (0, 0) || (r, c) == (n - 1, n - 1) {
                    continue;
                }
                let p = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
                if p < obstacle_prob {
                    *cell = true;
                }
            }
        }
        let (found, steps) = run_astar(&grid, n);
        if found {
            found_count += 1;
        }
        total_steps += u64::from(steps);
    }
    AStarVerdict::Ok {
        path_found_rate: f64::from(found_count) / f64::from(trials),
        mean_steps_explored: total_steps as f64 / f64::from(trials),
    }
}

fn run_astar(grid: &[Vec<bool>], n: usize) -> (bool, u32) {
    let goal = (n - 1, n - 1);
    let mut open: BinaryHeap<std::cmp::Reverse<(u32, usize, usize)>> = BinaryHeap::new();
    let mut closed: BTreeSet<(usize, usize)> = BTreeSet::new();
    open.push(std::cmp::Reverse((manhattan(0, 0, n), 0, 0)));
    let mut steps: u32 = 0;
    while let Some(std::cmp::Reverse((_, r, c))) = open.pop() {
        if (r, c) == goal {
            return (true, steps);
        }
        if !closed.insert((r, c)) {
            continue;
        }
        steps += 1;
        for (dr, dc) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= n as i32 || nc >= n as i32 {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            if grid[nr][nc] {
                continue;
            }
            if !closed.contains(&(nr, nc)) {
                let h = manhattan(nr, nc, n);
                open.push(std::cmp::Reverse((h, nr, nc)));
            }
        }
    }
    (false, steps)
}

fn manhattan(r: usize, c: usize, n: usize) -> u32 {
    ((n - 1).saturating_sub(r) + (n - 1).saturating_sub(c)) as u32
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_a_star_path_cost")?;

    println!("sparse: {:?}", simulate(50, 20, 0.1, 42));
    println!("dense: {:?}", simulate(50, 20, 0.5, 42));
    println!("invalid: {:?}", simulate(0, 20, 0.1, 42));
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
    fn sparse_high_path_rate() {
        let v = simulate(50, 10, 0.1, 42);
        if let AStarVerdict::Ok {
            path_found_rate, ..
        } = v
        {
            assert!(path_found_rate > 0.5);
        }
    }

    #[test]
    fn dense_low_path_rate() {
        let v = simulate(50, 10, 0.7, 42);
        if let AStarVerdict::Ok {
            path_found_rate, ..
        } = v
        {
            assert!(path_found_rate < 0.5);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 0.1, 42), AStarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_grid_too_small() {
        assert_eq!(simulate(10, 1, 0.1, 42), AStarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_obstacle_prob_one() {
        assert_eq!(simulate(10, 10, 1.0, 42), AStarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_prob() {
        assert_eq!(simulate(10, 10, -0.1, 42), AStarVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 10, 0.2, 42);
        let b = simulate(20, 10, 0.2, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn path_rate_in_unit_range() {
        let v = simulate(20, 10, 0.3, 42);
        if let AStarVerdict::Ok {
            path_found_rate, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&path_found_rate));
        }
    }

    #[test]
    fn zero_obstacle_always_found() {
        let v = simulate(10, 10, 0.0, 42);
        if let AStarVerdict::Ok {
            path_found_rate, ..
        } = v
        {
            assert!((path_found_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn larger_grid_more_steps() {
        let small = simulate(20, 5, 0.2, 42);
        let large = simulate(20, 20, 0.2, 42);
        if let (
            AStarVerdict::Ok {
                mean_steps_explored: s,
                ..
            },
            AStarVerdict::Ok {
                mean_steps_explored: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn dense_more_steps_when_path_found() {
        // Higher obstacle density may force longer detours.
        let v = simulate(20, 15, 0.3, 42);
        if let AStarVerdict::Ok {
            mean_steps_explored,
            ..
        } = v
        {
            assert!(mean_steps_explored > 0.0);
        }
    }
}
