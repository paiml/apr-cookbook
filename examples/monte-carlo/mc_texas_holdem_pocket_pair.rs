//! # Monte-Carlo Texas Hold'em Pocket-Pair Frequency
//!
//! Sample 2-card pocket hands from a 52-card deck without
//! replacement. Returns observed frequency of pocket-pairs vs the
//! theoretical 5.88% (= 13 × C(4,2) / C(52,2) = 78/1326).
//!
//! Demonstrates the **MC.155** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sklansky, The Theory of Poker (1987); standard
//!  combinatorial deck-deal analysis.
//!
//! Run with: cargo run --example mc_texas_holdem_pocket_pair
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HoldemVerdict {
    Ok {
        observed_pair_pct_x100: u32,
        theoretical_pct_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(hands: u32, seed: u64) -> HoldemVerdict {
    if hands < 1000 {
        return HoldemVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut pair_count = 0u32;
    for _ in 0..hands {
        let c1 = (lcg(&mut state) % 52) as u8;
        let mut c2 = (lcg(&mut state) % 52) as u8;
        // Resample if same card.
        while c2 == c1 {
            c2 = (lcg(&mut state) % 52) as u8;
        }
        // Card rank = card_id / 4 (0..=12); same-rank → pocket pair.
        if c1 / 4 == c2 / 4 {
            pair_count += 1;
        }
    }
    let observed = (pair_count as f64 / hands as f64) * 100.0 * 100.0;
    // Theoretical: 78/1326 ≈ 5.882% → 588 (×100).
    HoldemVerdict::Ok {
        observed_pair_pct_x100: observed as u32,
        theoretical_pct_x100: 588,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_texas_holdem_pocket_pair")?;

    println!("100k hands: {:?}", simulate(100_000, 42));
    println!("invalid: {:?}", simulate(50, 42));
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
    fn invalid_too_few_hands() {
        assert_eq!(simulate(50, 42), HoldemVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(5000, 42);
        let b = simulate(5000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn observed_near_theoretical() {
        let v = simulate(100_000, 42);
        if let HoldemVerdict::Ok {
            observed_pair_pct_x100,
            theoretical_pct_x100,
        } = v
        {
            // Allow ±1 percentage point at 100k hands.
            let diff = (observed_pair_pct_x100 as i32 - theoretical_pct_x100 as i32).abs();
            assert!(diff < 100);
        }
    }

    #[test]
    fn theoretical_constant() {
        let v = simulate(5000, 42);
        if let HoldemVerdict::Ok {
            theoretical_pct_x100,
            ..
        } = v
        {
            assert_eq!(theoretical_pct_x100, 588);
        }
    }

    #[test]
    fn observed_in_zero_to_100() {
        let v = simulate(5000, 42);
        if let HoldemVerdict::Ok {
            observed_pair_pct_x100,
            ..
        } = v
        {
            assert!(observed_pair_pct_x100 <= 10000);
        }
    }

    #[test]
    fn min_hands_accepted() {
        let v = simulate(1000, 42);
        assert!(matches!(v, HoldemVerdict::Ok { .. }));
    }

    #[test]
    fn many_hands_handled() {
        let v = simulate(1_000_000, 42);
        assert!(matches!(v, HoldemVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(2000, 42);
        let b = simulate(2000, 999);
        assert!(a != b);
    }

    #[test]
    fn observed_positive_at_large_n() {
        // With 100k hands, pair count should be > 0.
        let v = simulate(100_000, 42);
        if let HoldemVerdict::Ok {
            observed_pair_pct_x100,
            ..
        } = v
        {
            assert!(observed_pair_pct_x100 > 0);
        }
    }

    #[test]
    fn convergence_more_samples_tighter() {
        let small = simulate(1000, 42);
        let large = simulate(200_000, 42);
        if let (
            HoldemVerdict::Ok {
                observed_pair_pct_x100: s,
                ..
            },
            HoldemVerdict::Ok {
                observed_pair_pct_x100: l,
                ..
            },
        ) = (small, large)
        {
            let s_err = (s as i32 - 588).abs();
            let l_err = (l as i32 - 588).abs();
            assert!(l_err <= s_err);
        }
    }

    #[test]
    fn reasonable_pair_rate_at_low_n() {
        // 1k hands should still give pair rate in [3%, 9%] ~with high prob.
        let v = simulate(5000, 42);
        if let HoldemVerdict::Ok {
            observed_pair_pct_x100,
            ..
        } = v
        {
            assert!((300..=900).contains(&observed_pair_pct_x100));
        }
    }
}
