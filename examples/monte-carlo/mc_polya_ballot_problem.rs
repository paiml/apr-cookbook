//! # Monte-Carlo Bertrand-Polya Ballot Problem
//!
//! Sim the ballot problem: candidate A gets `a` votes, B gets `b`
//! votes (a > b). Probability that A is always strictly ahead during
//! count is (a-b)/(a+b). Returns observed probability vs theoretical.
//!
//! Demonstrates the **MC.193** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bertrand, Comptes Rendus 105 (1887); Polya
//!  enumeration arguments.
//!
//! Run with: cargo run --example mc_polya_ballot_problem
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BallotVerdict {
    Ok {
        observed_pct_x100: u32,
        theoretical_pct_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(votes_a: u32, votes_b: u32, trials: u32, seed: u64) -> BallotVerdict {
    if votes_a <= votes_b || votes_a == 0 || trials < 100 {
        return BallotVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let total = (votes_a + votes_b) as usize;
    let mut a_always_ahead = 0u32;
    for _ in 0..trials {
        // Build a random shuffle of 'a' A's and 'b' B's; check A always strictly ahead.
        let mut sequence: Vec<bool> = Vec::with_capacity(total);
        sequence.resize(votes_a as usize, true);
        sequence.resize(total, false);
        // Fisher-Yates
        for i in (1..sequence.len()).rev() {
            let j = (lcg(&mut state) as usize) % (i + 1);
            sequence.swap(i, j);
        }
        // Check A strictly ahead at all prefixes.
        let mut a_count = 0i32;
        let mut b_count = 0i32;
        let mut always_ahead = true;
        for is_a in &sequence {
            if *is_a {
                a_count += 1;
            } else {
                b_count += 1;
            }
            if a_count <= b_count {
                always_ahead = false;
                break;
            }
        }
        if always_ahead {
            a_always_ahead += 1;
        }
    }
    let observed = (a_always_ahead as f64 / trials as f64 * 10000.0) as u32;
    let theoretical =
        ((votes_a as f64 - votes_b as f64) / (votes_a as f64 + votes_b as f64) * 10000.0) as u32;
    BallotVerdict::Ok {
        observed_pct_x100: observed,
        theoretical_pct_x100: theoretical,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_polya_ballot_problem")?;

    println!("a=10, b=4: {:?}", simulate(10, 4, 5000, 42));
    println!("invalid: {:?}", simulate(5, 10, 100, 42));
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
    fn invalid_a_le_b() {
        assert_eq!(simulate(5, 10, 100, 42), BallotVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_a() {
        assert_eq!(simulate(0, 0, 100, 42), BallotVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 4, 50, 42), BallotVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 4, 1000, 42);
        let b = simulate(10, 4, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn observed_close_to_theoretical() {
        let v = simulate(10, 4, 50_000, 42);
        if let BallotVerdict::Ok {
            observed_pct_x100,
            theoretical_pct_x100,
        } = v
        {
            // (10-4)/14 = 0.4286 → 4286. Allow ±5%.
            let diff = (observed_pct_x100 as i32 - theoretical_pct_x100 as i32).abs();
            assert!(diff < 500);
        }
    }

    #[test]
    fn larger_lead_higher_prob() {
        let small = simulate(11, 9, 5000, 42);
        let large = simulate(11, 1, 5000, 42);
        if let (
            BallotVerdict::Ok {
                observed_pct_x100: s,
                ..
            },
            BallotVerdict::Ok {
                observed_pct_x100: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn theoretical_correct() {
        let v = simulate(10, 4, 1000, 42);
        if let BallotVerdict::Ok {
            theoretical_pct_x100,
            ..
        } = v
        {
            assert_eq!(theoretical_pct_x100, 4285);
        }
    }

    #[test]
    fn observed_in_zero_one() {
        let v = simulate(10, 4, 1000, 42);
        if let BallotVerdict::Ok {
            observed_pct_x100, ..
        } = v
        {
            assert!(observed_pct_x100 <= 10000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 1, 100, 42);
        assert!(matches!(v, BallotVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(10, 4, 10_000, 42);
        assert!(matches!(v, BallotVerdict::Ok { .. }));
    }

    #[test]
    fn full_lead_always_ahead() {
        // a > 0, b = 0 → A always ahead. Probability = 1.
        let v = simulate(10, 0, 5000, 42);
        if let BallotVerdict::Ok {
            observed_pct_x100, ..
        } = v
        {
            assert_eq!(observed_pct_x100, 10000);
        }
    }
}
