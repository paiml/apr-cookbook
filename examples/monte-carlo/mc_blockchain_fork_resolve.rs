//! # Monte-Carlo Blockchain Fork Resolution
//!
//! Sim two competing chains with random block creation rates. The
//! longest-chain rule selects the winner. Reports fork-resolution
//! time distribution.
//!
//! Demonstrates the **MC.127** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nakamoto, Bitcoin whitepaper §11 (longest-chain rule);
//!  Decker & Wattenhofer, P2P 2013 (fork analysis).
//!
//! Run with: cargo run --example mc_blockchain_fork_resolve
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ForkVerdict {
    Ok {
        avg_resolve_steps: f64,
        chain_a_wins: u32,
        chain_b_wins: u32,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, rate_a: f64, rate_b: f64, max_steps: u32, seed: u64) -> ForkVerdict {
    if trials == 0
        || max_steps == 0
        || !(0.0..=1.0).contains(&rate_a)
        || !(0.0..=1.0).contains(&rate_b)
    {
        return ForkVerdict::InvalidConfig;
    }
    let mut total_resolve_steps: u64 = 0;
    let mut chain_a_wins = 0u32;
    let mut chain_b_wins = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut len_a: u32 = 0;
        let mut len_b: u32 = 0;
        let mut resolved = false;
        for step in 0..max_steps {
            let r1 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r1 < rate_a {
                len_a += 1;
            }
            let r2 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r2 < rate_b {
                len_b += 1;
            }
            // Resolved when one chain leads by >= 6 (Nakamoto's "6 confirmations").
            if (len_a as i32 - len_b as i32).abs() >= 6 {
                total_resolve_steps += u64::from(step + 1);
                resolved = true;
                if len_a > len_b {
                    chain_a_wins += 1;
                } else {
                    chain_b_wins += 1;
                }
                break;
            }
        }
        if !resolved {
            total_resolve_steps += u64::from(max_steps);
        }
    }
    ForkVerdict::Ok {
        avg_resolve_steps: total_resolve_steps as f64 / f64::from(trials),
        chain_a_wins,
        chain_b_wins,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_blockchain_fork_resolve")?;

    println!("equal rates: {:?}", simulate(500, 0.1, 0.1, 1000, 42));
    println!("a dominant: {:?}", simulate(500, 0.3, 0.05, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.1, 0.1, 1000, 42));
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
    fn faster_chain_wins_more() {
        let v = simulate(500, 0.3, 0.05, 1000, 42);
        if let ForkVerdict::Ok {
            chain_a_wins,
            chain_b_wins,
            ..
        } = v
        {
            assert!(chain_a_wins > chain_b_wins);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 0.1, 0.1, 1000, 42), ForkVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(100, 0.1, 0.1, 0, 42), ForkVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_rate_out_of_range() {
        assert_eq!(
            simulate(100, 1.5, 0.1, 1000, 42),
            ForkVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 0.1, 0.1, 1000, 42);
        let b = simulate(100, 0.1, 0.1, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn wins_le_trials() {
        let v = simulate(500, 0.1, 0.1, 1000, 42);
        if let ForkVerdict::Ok {
            chain_a_wins,
            chain_b_wins,
            ..
        } = v
        {
            assert!(chain_a_wins + chain_b_wins <= 500);
        }
    }

    #[test]
    fn equal_rates_balanced() {
        let v = simulate(500, 0.2, 0.2, 2000, 42);
        if let ForkVerdict::Ok {
            chain_a_wins,
            chain_b_wins,
            ..
        } = v
        {
            let diff = chain_a_wins.abs_diff(chain_b_wins);
            assert!(diff < chain_a_wins.max(chain_b_wins));
        }
    }

    #[test]
    fn higher_rates_faster_resolve() {
        let lo = simulate(500, 0.05, 0.05, 5000, 42);
        let hi = simulate(500, 0.5, 0.5, 5000, 42);
        if let (
            ForkVerdict::Ok {
                avg_resolve_steps: l,
                ..
            },
            ForkVerdict::Ok {
                avg_resolve_steps: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h < l);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 0.1, 0.1, 1000, 42);
        if let ForkVerdict::Ok {
            avg_resolve_steps, ..
        } = v
        {
            assert!(avg_resolve_steps.is_finite());
        }
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(2000, 0.1, 0.1, 1000, 42);
        assert!(matches!(v, ForkVerdict::Ok { .. }));
    }

    #[test]
    fn zero_rates_no_resolve() {
        let v = simulate(50, 0.0, 0.0, 100, 42);
        if let ForkVerdict::Ok {
            chain_a_wins,
            chain_b_wins,
            ..
        } = v
        {
            assert_eq!(chain_a_wins, 0);
            assert_eq!(chain_b_wins, 0);
        }
    }
}
