//! # Monte-Carlo Self-Avoiding Walk
//!
//! Sim a self-avoiding random walk on a 2D grid: at each step, choose
//! a random unvisited neighbor; if all neighbors visited, walk stops.
//! Returns mean walk length and successful-completion rate.
//!
//! Demonstrates the **MC.188** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hammersley, "On the rate of convergence to the
//!  connective constant" (1957); polymer chain models.
//!
//! Run with: cargo run --example mc_self_avoiding_walk
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum SawVerdict {
    Ok {
        mean_length: u32,
        completion_rate_x1000: u32,
    },
    InvalidConfig,
}

pub fn simulate(target_length: u32, trials: u32, seed: u64) -> SawVerdict {
    if target_length < 5 || trials < 100 {
        return SawVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_length: u64 = 0;
    let mut completed = 0u32;
    for _ in 0..trials {
        let mut visited: HashSet<(i32, i32)> = HashSet::new();
        let mut x = 0i32;
        let mut y = 0i32;
        visited.insert((x, y));
        let mut length = 0u32;
        for _ in 0..target_length {
            // Choose random unvisited neighbor.
            let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            let mut available: Vec<(i32, i32)> = dirs
                .iter()
                .map(|(dx, dy)| (x + dx, y + dy))
                .filter(|p| !visited.contains(p))
                .collect();
            if available.is_empty() {
                break;
            }
            let idx = (lcg(&mut state) as usize) % available.len();
            let (nx, ny) = available.remove(idx);
            visited.insert((nx, ny));
            x = nx;
            y = ny;
            length += 1;
        }
        total_length += length as u64;
        if length == target_length {
            completed += 1;
        }
    }
    SawVerdict::Ok {
        mean_length: (total_length / trials as u64) as u32,
        completion_rate_x1000: ((completed as f64 / trials as f64) * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_self_avoiding_walk")?;

    println!("len=10: {:?}", simulate(10, 1000, 42));
    println!("len=50: {:?}", simulate(50, 1000, 42));
    println!("invalid: {:?}", simulate(2, 100, 42));
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
    fn invalid_too_short_target() {
        assert_eq!(simulate(2, 100, 42), SawVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 42), SawVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 100, 42);
        let b = simulate(10, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_walks_lower_completion() {
        let short = simulate(10, 500, 42);
        let long = simulate(50, 500, 42);
        if let (
            SawVerdict::Ok {
                completion_rate_x1000: s,
                ..
            },
            SawVerdict::Ok {
                completion_rate_x1000: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l <= s);
        }
    }

    #[test]
    fn mean_length_le_target() {
        let v = simulate(20, 500, 42);
        if let SawVerdict::Ok { mean_length, .. } = v {
            assert!(mean_length <= 20);
        }
    }

    #[test]
    fn completion_rate_in_zero_one() {
        let v = simulate(10, 500, 42);
        if let SawVerdict::Ok {
            completion_rate_x1000,
            ..
        } = v
        {
            assert!(completion_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 100, 42);
        assert!(matches!(v, SawVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(20, 10_000, 42);
        assert!(matches!(v, SawVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(10, 500, 42);
        let b = simulate(10, 500, 999);
        assert!(a != b);
    }

    #[test]
    fn short_walks_high_completion() {
        // Length 5 should usually complete (low chance of trapping).
        let v = simulate(5, 1000, 42);
        if let SawVerdict::Ok {
            completion_rate_x1000,
            ..
        } = v
        {
            assert!(completion_rate_x1000 > 800);
        }
    }

    #[test]
    fn finite_outcomes() {
        let v = simulate(20, 100, 42);
        if let SawVerdict::Ok {
            mean_length,
            completion_rate_x1000,
        } = v
        {
            assert!(mean_length < u32::MAX);
            assert!(completion_rate_x1000 <= 1000);
        }
    }
}
