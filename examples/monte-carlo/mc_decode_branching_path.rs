//! # Monte-Carlo Decode-Branching Path
//!
//! Simulate beam-search decode: at each step, K candidate paths
//! survive with probability proportional to their score. Returns
//! observed survivor depth and mean branching factor.
//!
//! Demonstrates the **MC.15** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Beam search analysis (Vijayakumar et al. 2018).
//!
//! Run with: cargo run --example mc_decode_branching_path
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BranchVerdict {
    Ok {
        mean_survivor_depth: f64,
        max_survivor_depth: u32,
        early_termination_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    beam_width: u32,
    max_depth: u32,
    survive_prob: f64,
    runs: u32,
    seed: u64,
) -> BranchVerdict {
    if beam_width == 0
        || max_depth == 0
        || runs == 0
        || !survive_prob.is_finite()
        || !(0.0..=1.0).contains(&survive_prob)
    {
        return BranchVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_depth: u64 = 0;
    let mut max_v: u32 = 0;
    let mut early: u32 = 0;
    for _ in 0..runs {
        let mut survivors = beam_width;
        let mut depth: u32 = 0;
        for _ in 0..max_depth {
            if survivors == 0 {
                break;
            }
            let mut still_alive = 0u32;
            for _ in 0..survivors {
                if unit(&mut rng_state) < survive_prob {
                    still_alive += 1;
                }
            }
            survivors = still_alive;
            depth += 1;
            if survivors == 0 {
                early += 1;
                break;
            }
        }
        total_depth += u64::from(depth);
        max_v = max_v.max(depth);
    }
    let mean_survivor_depth = total_depth as f64 / f64::from(runs);
    let early_termination_pct = (f64::from(early) / f64::from(runs)) * 100.0;
    BranchVerdict::Ok {
        mean_survivor_depth,
        max_survivor_depth: max_v,
        early_termination_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_decode_branching_path")?;

    println!("high survival: {:?}", simulate(8, 20, 0.95, 1000, 42));
    println!("low survival: {:?}", simulate(8, 20, 0.50, 1000, 42));
    println!("invalid: {:?}", simulate(0, 20, 0.5, 1000, 42));
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
    fn high_survival_full_depth() {
        let v = simulate(8, 20, 0.95, 1000, 42);
        if let BranchVerdict::Ok {
            mean_survivor_depth,
            ..
        } = v
        {
            assert!(mean_survivor_depth > 15.0);
        }
    }

    #[test]
    fn low_survival_early_term() {
        let v = simulate(2, 20, 0.30, 1000, 42);
        if let BranchVerdict::Ok {
            early_termination_pct,
            ..
        } = v
        {
            assert!(early_termination_pct > 50.0);
        }
    }

    #[test]
    fn invalid_zero_beam() {
        assert_eq!(simulate(0, 20, 0.5, 1000, 42), BranchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_depth() {
        assert_eq!(simulate(8, 0, 0.5, 1000, 42), BranchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_runs() {
        assert_eq!(simulate(8, 20, 0.5, 0, 42), BranchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_p() {
        assert_eq!(
            simulate(8, 20, -0.1, 1000, 42),
            BranchVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(8, 20, f64::NAN, 1000, 42),
            BranchVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(8, 20, 0.7, 100, 42);
        let b = simulate(8, 20, 0.7, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_bounded_by_depth_limit() {
        let v = simulate(8, 20, 0.99, 100, 42);
        if let BranchVerdict::Ok {
            max_survivor_depth, ..
        } = v
        {
            assert!(max_survivor_depth <= 20);
        }
    }

    #[test]
    fn higher_survival_higher_mean_depth() {
        let lo = simulate(8, 20, 0.30, 500, 42);
        let hi = simulate(8, 20, 0.95, 500, 42);
        if let (
            BranchVerdict::Ok {
                mean_survivor_depth: a,
                ..
            },
            BranchVerdict::Ok {
                mean_survivor_depth: b,
                ..
            },
        ) = (lo, hi)
        {
            assert!(b > a);
        }
    }
}
