//! # Monte-Carlo Spinning Wheel Payout
//!
//! Sim a roulette-style wheel: N pockets each with a payout multiple
//! of the bet. Returns mean payout per spin (×100) and house-edge
//! estimate (positive means house wins).
//!
//! Demonstrates the **MC.154** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Thorp, Beat the Dealer (1962); Epstein, Theory of
//!  Gambling and Statistical Logic ch. 6 (1967).
//!
//! Run with: cargo run --example mc_spinning_wheel_payout
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WheelVerdict {
    Ok {
        mean_payout_x100: i32,
        house_edge_x100: i32,
    },
    InvalidConfig,
}

pub fn simulate(payouts: &[i32], spins: u32, seed: u64) -> WheelVerdict {
    if payouts.is_empty() || spins < 100 {
        return WheelVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total: i64 = 0;
    for _ in 0..spins {
        let idx = (lcg(&mut state) as usize) % payouts.len();
        total += payouts[idx] as i64;
    }
    let mean = total as f64 / spins as f64;
    let edge = -mean; // house wins when mean payout to player is negative
    WheelVerdict::Ok {
        mean_payout_x100: (mean * 100.0) as i32,
        house_edge_x100: (edge * 100.0) as i32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_spinning_wheel_payout")?;

    // 38-pocket American roulette: 2 zeros (-1), 36 numbers (-1 + 35 if hit)
    // For even-money bet on red (18 pockets), win=+1, lose=-1, with
    // 18 wins, 20 losses out of 38 → expected payout -2/38 ≈ -0.0526.
    let red_bet = vec![1i32; 18]
        .into_iter()
        .chain(vec![-1i32; 20])
        .collect::<Vec<_>>();
    println!("red: {:?}", simulate(&red_bet, 100_000, 42));
    println!("invalid: {:?}", simulate(&[], 100, 42));
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
    fn invalid_empty_payouts() {
        assert_eq!(simulate(&[], 100, 42), WheelVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_spins() {
        assert_eq!(simulate(&[1, -1], 50, 42), WheelVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(&[1, -1], 1000, 42);
        let b = simulate(&[1, -1], 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fair_coin_zero_mean() {
        // Equal +1/-1 payouts → fair, mean ~0, edge ~0.
        let v = simulate(&[1, -1], 100_000, 42);
        if let WheelVerdict::Ok {
            mean_payout_x100, ..
        } = v
        {
            assert!(mean_payout_x100.abs() < 5);
        }
    }

    #[test]
    fn house_edge_negates_mean() {
        let v = simulate(&[1, -1, -1, -1], 10_000, 42);
        if let WheelVerdict::Ok {
            mean_payout_x100,
            house_edge_x100,
        } = v
        {
            assert_eq!(house_edge_x100, -mean_payout_x100);
        }
    }

    #[test]
    fn american_roulette_red_negative_edge() {
        // 18 reds (+1) vs 20 non-reds (-1) → expected -2/38 ≈ -0.0526.
        let red_bet: Vec<i32> = vec![1i32; 18].into_iter().chain(vec![-1i32; 20]).collect();
        let v = simulate(&red_bet, 100_000, 42);
        if let WheelVerdict::Ok {
            mean_payout_x100, ..
        } = v
        {
            assert!(mean_payout_x100 < 0);
            assert!((-15..=-1).contains(&mean_payout_x100));
        }
    }

    #[test]
    fn always_win_positive_mean() {
        let v = simulate(&[1, 1, 1], 1000, 42);
        if let WheelVerdict::Ok {
            mean_payout_x100, ..
        } = v
        {
            assert_eq!(mean_payout_x100, 100);
        }
    }

    #[test]
    fn always_lose_negative_mean() {
        let v = simulate(&[-1, -1], 1000, 42);
        if let WheelVerdict::Ok {
            mean_payout_x100, ..
        } = v
        {
            assert_eq!(mean_payout_x100, -100);
        }
    }

    #[test]
    fn min_spins_accepted() {
        let v = simulate(&[1, -1], 100, 42);
        assert!(matches!(v, WheelVerdict::Ok { .. }));
    }

    #[test]
    fn many_spins_handled() {
        let v = simulate(&[1, -1], 1_000_000, 42);
        assert!(matches!(v, WheelVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(&[1, -1], 500, 42);
        let b = simulate(&[1, -1], 500, 999);
        assert!(a != b);
    }
}
