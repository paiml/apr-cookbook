//! # Monte-Carlo Kelly Criterion Bet Sizing
//!
//! Sim a sequence of independent bets where the Kelly-optimal fraction
//! is bet on each. Returns final bankroll multiple (×100) after the
//! sequence vs naive flat-stake comparison.
//!
//! Demonstrates the **MC.167** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kelly, "A New Interpretation of Information Rate"
//!  Bell Labs Tech. J. (1956); Thorp's Beat the Dealer (1962).
//!
//! Run with: cargo run --example mc_kelly_criterion_bet_size
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KellyVerdict {
    Ok {
        kelly_final_x100: u32,
        flat_final_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(win_prob_x100: u32, win_payout_x100: u32, bets: u32, seed: u64) -> KellyVerdict {
    if !(1..=99).contains(&win_prob_x100) || win_payout_x100 == 0 || bets < 100 {
        return KellyVerdict::InvalidConfig;
    }
    let p = win_prob_x100 as f64 / 100.0;
    let q = 1.0 - p;
    let b = win_payout_x100 as f64 / 100.0;
    // Kelly fraction f* = (bp - q) / b
    let kelly_frac = ((b * p - q) / b).clamp(0.0, 1.0);
    let flat_frac = 0.05; // arbitrary flat stake for comparison
    let mut state = seed | 1;
    let mut bankroll_kelly = 1.0f64;
    let mut bankroll_flat = 1.0f64;
    for _ in 0..bets {
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let win = r < p;
        if win {
            bankroll_kelly *= 1.0 + kelly_frac * b;
            bankroll_flat *= 1.0 + flat_frac * b;
        } else {
            bankroll_kelly *= 1.0 - kelly_frac;
            bankroll_flat *= 1.0 - flat_frac;
        }
    }
    KellyVerdict::Ok {
        kelly_final_x100: (bankroll_kelly.max(0.0) * 100.0) as u32,
        flat_final_x100: (bankroll_flat.max(0.0) * 100.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kelly_criterion_bet_size")?;

    println!("60% win, 1:1 payout: {:?}", simulate(60, 100, 1000, 42));
    println!("invalid: {:?}", simulate(0, 100, 1000, 42));
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
    fn invalid_zero_win_prob() {
        assert_eq!(simulate(0, 100, 1000, 42), KellyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_win_prob() {
        assert_eq!(simulate(100, 100, 1000, 42), KellyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_payout() {
        assert_eq!(simulate(60, 0, 1000, 42), KellyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_bets() {
        assert_eq!(simulate(60, 100, 50, 42), KellyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(60, 100, 500, 42);
        let b = simulate(60, 100, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn favorable_bet_grows_bankroll() {
        let v = simulate(60, 100, 5000, 42);
        if let KellyVerdict::Ok {
            kelly_final_x100, ..
        } = v
        {
            // 60% win at 1:1 → positive expectation, Kelly grows long-run.
            assert!(kelly_final_x100 > 100);
        }
    }

    #[test]
    fn unfavorable_bet_shrinks_bankroll() {
        let v = simulate(40, 100, 5000, 42);
        if let KellyVerdict::Ok {
            kelly_final_x100, ..
        } = v
        {
            // Negative expectation → Kelly returns 0 → no bet → bankroll preserved at 100.
            assert_eq!(kelly_final_x100, 100);
        }
    }

    #[test]
    fn flat_returned() {
        let v = simulate(60, 100, 1000, 42);
        if let KellyVerdict::Ok {
            flat_final_x100, ..
        } = v
        {
            assert!(flat_final_x100 < u32::MAX);
        }
    }

    #[test]
    fn min_bets_accepted() {
        let v = simulate(60, 100, 100, 42);
        assert!(matches!(v, KellyVerdict::Ok { .. }));
    }

    #[test]
    fn many_bets_handled() {
        let v = simulate(60, 100, 50_000, 42);
        assert!(matches!(v, KellyVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(60, 100, 500, 42);
        let b = simulate(60, 100, 500, 999);
        assert!(a != b);
    }
}
