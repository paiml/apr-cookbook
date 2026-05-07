//! # Monte-Carlo 100 Prisoners and 100 Boxes
//!
//! Sim the classic problem: N prisoners each must find their number
//! among N boxes, each opening at most N/2. Strategy: open the box
//! marked with your own number, then follow the cycle. If all
//! prisoners succeed → group escapes.
//!
//! Demonstrates the **MC.161** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gál & Miltersen, "100 prisoners problem" Theory of
//!  Computing 4(1) (2008); Curtin & Warshauer, MAM J. (2007).
//!
//! Run with: cargo run --example mc_prisoner_box_strategy
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PrisonerVerdict {
    Ok {
        success_rate_x1000: u32,
        prisoners: u32,
    },
    InvalidConfig,
}

pub fn simulate(prisoners: u32, trials: u32, seed: u64) -> PrisonerVerdict {
    if prisoners < 4 || prisoners % 2 != 0 || trials < 100 {
        return PrisonerVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let half = (prisoners / 2) as usize;
    let n = prisoners as usize;
    let mut group_successes = 0u32;
    for _ in 0..trials {
        // Random permutation of boxes 0..n containing labels 0..n.
        let mut boxes: Vec<u32> = (0..prisoners).collect();
        for i in (1..boxes.len()).rev() {
            let j = (lcg(&mut state) as usize) % (i + 1);
            boxes.swap(i, j);
        }
        let mut all_succeed = true;
        for prisoner in 0..n {
            let mut idx = prisoner;
            let mut found = false;
            for _ in 0..half {
                let label = boxes[idx] as usize;
                if label == prisoner {
                    found = true;
                    break;
                }
                idx = label;
            }
            if !found {
                all_succeed = false;
                break;
            }
        }
        if all_succeed {
            group_successes += 1;
        }
    }
    let rate = (group_successes as f64 / trials as f64) * 1000.0;
    PrisonerVerdict::Ok {
        success_rate_x1000: rate as u32,
        prisoners,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_prisoner_box_strategy")?;

    println!("n=100: {:?}", simulate(100, 5000, 42));
    println!("invalid: {:?}", simulate(3, 100, 42));
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
    fn invalid_too_few_prisoners() {
        assert_eq!(simulate(3, 100, 42), PrisonerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_odd_prisoners() {
        assert_eq!(simulate(5, 100, 42), PrisonerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 42), PrisonerVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 1000, 42);
        let b = simulate(20, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn success_rate_near_31_percent_for_100() {
        // Theoretical limit: 1 - ln(2) ≈ 0.3069 → 307.
        let v = simulate(100, 20_000, 42);
        if let PrisonerVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!((250..=380).contains(&success_rate_x1000));
        }
    }

    #[test]
    fn rate_in_zero_one() {
        let v = simulate(20, 1000, 42);
        if let PrisonerVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!(success_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn prisoners_returned() {
        let v = simulate(50, 1000, 42);
        if let PrisonerVerdict::Ok { prisoners, .. } = v {
            assert_eq!(prisoners, 50);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(4, 100, 42);
        assert!(matches!(v, PrisonerVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(20, 10_000, 42);
        assert!(matches!(v, PrisonerVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(20, 500, 42);
        let b = simulate(20, 500, 999);
        assert!(a != b);
    }

    #[test]
    fn rate_converges_to_one_minus_ln2() {
        // For very large n, rate → 1 - ln(2) ≈ 0.307.
        let v = simulate(200, 10_000, 42);
        if let PrisonerVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!((230..=380).contains(&success_rate_x1000));
        }
    }
}
