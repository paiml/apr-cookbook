//! # Monte-Carlo St. Petersburg Paradox
//!
//! Sim the St. Petersburg game: a fair coin is flipped until heads
//! appears on flip k; payout is 2^k. Theoretical EV is infinite, but
//! with bounded house cap the empirical mean converges.
//!
//! Demonstrates the **MC.145** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bernoulli, Specimen Theoriae Novae de Mensura Sortis
//!  (1738); Menger 1934 cap-rule analysis.
//!
//! Run with: cargo run --example mc_st_petersburg_paradox
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PetersburgVerdict {
    Ok {
        empirical_mean_x100: u32,
        max_payout_observed: u64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, max_payout_cap: u32, seed: u64) -> PetersburgVerdict {
    if trials < 100 || max_payout_cap == 0 {
        return PetersburgVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total: u64 = 0;
    let mut max_obs: u64 = 0;
    for _ in 0..trials {
        let mut k: u32 = 1;
        // Flip until heads (high bit set) - cap at 32 for practical termination
        while k < 32 {
            let bit = lcg(&mut state) >> 32 & 1;
            if bit == 0 {
                break;
            }
            k += 1;
        }
        let payout: u64 = 1u64 << k.min(31);
        let capped = payout.min(max_payout_cap as u64);
        total += capped;
        if capped > max_obs {
            max_obs = capped;
        }
    }
    let mean = total / trials as u64;
    PetersburgVerdict::Ok {
        empirical_mean_x100: (mean as u32).saturating_mul(100),
        max_payout_observed: max_obs,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_st_petersburg_paradox")?;

    println!("cap-1024: {:?}", simulate(10_000, 1024, 42));
    println!("cap-1M: {:?}", simulate(10_000, 1_000_000, 42));
    println!("invalid: {:?}", simulate(50, 100, 42));
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
    fn invalid_too_few_trials() {
        assert_eq!(simulate(50, 100, 42), PetersburgVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_cap() {
        assert_eq!(simulate(1000, 0, 42), PetersburgVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 1024, 42);
        let b = simulate(500, 1024, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_payout_le_cap() {
        let v = simulate(10_000, 1024, 42);
        if let PetersburgVerdict::Ok {
            max_payout_observed,
            ..
        } = v
        {
            assert!(max_payout_observed <= 1024);
        }
    }

    #[test]
    fn bigger_cap_bigger_mean() {
        let small = simulate(10_000, 16, 42);
        let big = simulate(10_000, 1_000_000, 42);
        if let (
            PetersburgVerdict::Ok {
                empirical_mean_x100: s,
                ..
            },
            PetersburgVerdict::Ok {
                empirical_mean_x100: b,
                ..
            },
        ) = (small, big)
        {
            assert!(b >= s);
        }
    }

    #[test]
    fn empirical_mean_finite() {
        let v = simulate(1000, 1024, 42);
        if let PetersburgVerdict::Ok {
            empirical_mean_x100,
            ..
        } = v
        {
            assert!(empirical_mean_x100 < u32::MAX);
        }
    }

    #[test]
    fn min_payout_ge_2() {
        // Minimum payout = 2^1 = 2 (heads on first flip)
        let v = simulate(1000, 1024, 42);
        if let PetersburgVerdict::Ok {
            empirical_mean_x100,
            ..
        } = v
        {
            // mean × 100 ≥ 2 × 100 = 200
            assert!(empirical_mean_x100 >= 200);
        }
    }

    #[test]
    fn small_cap_caps_payout() {
        let v = simulate(1000, 4, 42);
        if let PetersburgVerdict::Ok {
            max_payout_observed,
            ..
        } = v
        {
            assert!(max_payout_observed <= 4);
        }
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100_000, 1024, 42);
        assert!(matches!(v, PetersburgVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcome() {
        let a = simulate(500, 1024, 42);
        let b = simulate(500, 1024, 999);
        assert!(a != b);
    }

    #[test]
    fn min_trials_accepted() {
        let v = simulate(100, 1024, 42);
        assert!(matches!(v, PetersburgVerdict::Ok { .. }));
    }
}
