//! # Monte-Carlo 2D Grid Random Walk Diffusion
//!
//! Sim a random walk on a 2D integer grid (4 directions). Returns
//! mean-squared displacement and the maximum cell visited.
//!
//! Demonstrates the **MC.165** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pólya (1921) recurrence theorem for 2D random walks;
//!  Einstein (1905) diffusion: ⟨r²⟩ ∝ N for unbiased walk.
//!
//! Run with: cargo run --example mc_grid_walk_2d_diffusion
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GridWalkVerdict {
    Ok {
        mean_sq_disp: u32,
        max_distance: u32,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, trials: u32, seed: u64) -> GridWalkVerdict {
    if steps < 10 || trials < 50 {
        return GridWalkVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_sq_disp: u64 = 0;
    let mut max_dist = 0u32;
    for _ in 0..trials {
        let mut x = 0i32;
        let mut y = 0i32;
        for _ in 0..steps {
            let dir = (lcg(&mut state) >> 32) % 4;
            match dir {
                0 => x += 1,
                1 => x -= 1,
                2 => y += 1,
                _ => y -= 1,
            }
        }
        let sq = (x * x + y * y) as u64;
        total_sq_disp += sq;
        let dist = ((x * x + y * y) as f64).sqrt() as u32;
        if dist > max_dist {
            max_dist = dist;
        }
    }
    GridWalkVerdict::Ok {
        mean_sq_disp: (total_sq_disp / trials as u64) as u32,
        max_distance: max_dist,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_grid_walk_2d_diffusion")?;

    println!("100 steps: {:?}", simulate(100, 1000, 42));
    println!("invalid: {:?}", simulate(5, 50, 42));
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
    fn invalid_too_few_steps() {
        assert_eq!(simulate(5, 50, 42), GridWalkVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(100, 10, 42), GridWalkVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_sq_near_n_for_unbiased_walk() {
        // ⟨r²⟩ ≈ N for 2D unbiased walk.
        let v = simulate(100, 5000, 42);
        if let GridWalkVerdict::Ok { mean_sq_disp, .. } = v {
            assert!((50..=150).contains(&mean_sq_disp));
        }
    }

    #[test]
    fn max_distance_le_steps() {
        let v = simulate(100, 100, 42);
        if let GridWalkVerdict::Ok { max_distance, .. } = v {
            assert!(max_distance <= 100);
        }
    }

    #[test]
    fn longer_walks_more_displacement() {
        let short = simulate(100, 1000, 42);
        let long = simulate(1000, 1000, 42);
        if let (
            GridWalkVerdict::Ok {
                mean_sq_disp: s, ..
            },
            GridWalkVerdict::Ok {
                mean_sq_disp: l, ..
            },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 50, 42);
        assert!(matches!(v, GridWalkVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100, 10_000, 42);
        assert!(matches!(v, GridWalkVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 999);
        assert!(a != b);
    }

    #[test]
    fn max_distance_at_least_one() {
        // Even single trial should move at least one step.
        let v = simulate(100, 100, 42);
        if let GridWalkVerdict::Ok { max_distance, .. } = v {
            assert!(max_distance >= 1);
        }
    }

    #[test]
    fn finite_msd() {
        let v = simulate(100, 100, 42);
        if let GridWalkVerdict::Ok { mean_sq_disp, .. } = v {
            assert!(mean_sq_disp < u32::MAX);
        }
    }
}
