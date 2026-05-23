//! # Monte-Carlo Kingman Coalescent Tree Sample
//!
//! Sim Kingman's coalescent: starting with n lineages, merge two
//! random lineages at exponential intervals until 1 remains. Returns
//! total tree height and total branch length.
//!
//! Demonstrates the **MC.158** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kingman, "The Coalescent" Stochastic Processes (1982);
//!  Hudson coalescent simulator (1990).
//!
//! Run with: cargo run --example mc_coalescent_tree_sample
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CoalescentVerdict {
    Ok {
        tree_height_x1000: u32,
        total_branch_length_x1000: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_lineages: u32, seed: u64) -> CoalescentVerdict {
    if n_lineages < 2 {
        return CoalescentVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut k = n_lineages;
    let mut total_height = 0.0f64;
    let mut total_branch = 0.0f64;
    while k > 1 {
        // Time to next coalescence ~ Exp(C(k,2)) where rate = k(k-1)/2.
        let rate = (k as f64) * ((k - 1) as f64) / 2.0;
        let raw = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let u = raw.max(1e-10);
        let dt = -(1.0 - u).ln() / rate;
        total_height += dt;
        total_branch += dt * (k as f64);
        k -= 1;
    }
    CoalescentVerdict::Ok {
        tree_height_x1000: (total_height * 1000.0) as u32,
        total_branch_length_x1000: (total_branch * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_coalescent_tree_sample")?;

    println!("n=10: {:?}", simulate(10, 42));
    println!("invalid: {:?}", simulate(1, 42));
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
    fn invalid_too_few_lineages() {
        assert_eq!(simulate(1, 42), CoalescentVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 42);
        let b = simulate(10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn tree_height_positive() {
        let v = simulate(10, 42);
        if let CoalescentVerdict::Ok {
            tree_height_x1000, ..
        } = v
        {
            assert!(tree_height_x1000 > 0);
        }
    }

    #[test]
    fn branch_length_at_least_height() {
        // Total branch = sum(dt * k) ≥ tree_height (since k ≥ 1).
        let v = simulate(10, 42);
        if let CoalescentVerdict::Ok {
            tree_height_x1000,
            total_branch_length_x1000,
        } = v
        {
            assert!(total_branch_length_x1000 >= tree_height_x1000);
        }
    }

    #[test]
    fn larger_n_more_branch_length() {
        let small = simulate(5, 42);
        let large = simulate(50, 42);
        if let (
            CoalescentVerdict::Ok {
                total_branch_length_x1000: s,
                ..
            },
            CoalescentVerdict::Ok {
                total_branch_length_x1000: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn min_lineages_accepted() {
        let v = simulate(2, 42);
        assert!(matches!(v, CoalescentVerdict::Ok { .. }));
    }

    #[test]
    fn many_lineages_handled() {
        let v = simulate(1000, 42);
        assert!(matches!(v, CoalescentVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(10, 42);
        let b = simulate(10, 999);
        assert!(a != b);
    }

    #[test]
    fn n_2_branch_double_height() {
        // n=2: single coalescence at time t; branch = 2t.
        let v = simulate(2, 42);
        if let CoalescentVerdict::Ok {
            tree_height_x1000,
            total_branch_length_x1000,
        } = v
        {
            // total_branch = 2 * tree_height (within rounding).
            let diff = total_branch_length_x1000 as i64 - 2 * tree_height_x1000 as i64;
            assert!(diff.abs() <= 1);
        }
    }

    #[test]
    fn expected_height_near_2() {
        // E[T_MRCA] = 2(1 - 1/n) for Kingman. n=100 → ~1.98.
        let mut sum = 0u64;
        for s in 0..100 {
            if let CoalescentVerdict::Ok {
                tree_height_x1000, ..
            } = simulate(100, s)
            {
                sum += tree_height_x1000 as u64;
            }
        }
        let mean = sum / 100;
        // Allow wide tolerance since coalescent has high variance.
        assert!((1000..=4000).contains(&mean));
    }

    #[test]
    fn finite_tree_height() {
        let v = simulate(10, 42);
        if let CoalescentVerdict::Ok {
            tree_height_x1000, ..
        } = v
        {
            assert!(tree_height_x1000 < u32::MAX);
        }
    }
}
