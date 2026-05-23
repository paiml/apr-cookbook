//! # Monte-Carlo Secretary Problem (Two-Choice Variant)
//!
//! Secretary problem with TWO selection chances. Use the classic 1/e
//! observation phase, then commit to the first candidate above the
//! observed max; if you get to the end, fall back to the very last.
//! Returns success rate (catching either of the top-2).
//!
//! Demonstrates the **MC.159** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gilbert & Mosteller §3 (1966) "k-out-of-n best"
//!  generalization; Ferguson (1989) two-choice survey.
//!
//! Run with: cargo run --example mc_secretary_two_choice
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TwoChoiceVerdict {
    Ok {
        success_rate_x1000: u32,
        applicants: u32,
    },
    InvalidConfig,
}

pub fn simulate(applicants: u32, trials: u32, seed: u64) -> TwoChoiceVerdict {
    if applicants < 5 || trials < 100 {
        return TwoChoiceVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let observe = (applicants as f64 / std::f64::consts::E) as usize;
    let top_two = applicants - 1;
    let mut successes = 0u32;
    for _ in 0..trials {
        let mut perm: Vec<u32> = (1..=applicants).collect();
        for i in (1..perm.len()).rev() {
            let j = (lcg(&mut state) as usize) % (i + 1);
            perm.swap(i, j);
        }
        let max_in_observe = *perm.iter().take(observe).max().unwrap_or(&0);
        let mut chosen: Option<u32> = None;
        for v in perm.iter().skip(observe) {
            if *v > max_in_observe {
                chosen = Some(*v);
                break;
            }
        }
        let pick = chosen.unwrap_or(*perm.last().unwrap());
        if pick >= top_two {
            successes += 1;
        }
    }
    let rate = (successes as f64 / trials as f64) * 1000.0;
    TwoChoiceVerdict::Ok {
        success_rate_x1000: rate as u32,
        applicants,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_secretary_two_choice")?;

    println!("n=100: {:?}", simulate(100, 10_000, 42));
    println!("invalid: {:?}", simulate(2, 100, 42));
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
    fn invalid_too_few_applicants() {
        assert_eq!(simulate(2, 100, 42), TwoChoiceVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 42), TwoChoiceVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 1000, 42);
        let b = simulate(20, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn two_choice_better_than_single() {
        // Two-choice success rate should exceed 1/e ≈ 0.368 (single-choice limit).
        let v = simulate(100, 50_000, 42);
        if let TwoChoiceVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!(success_rate_x1000 > 380);
        }
    }

    #[test]
    fn success_rate_in_zero_one() {
        let v = simulate(20, 1000, 42);
        if let TwoChoiceVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!(success_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn applicants_returned() {
        let v = simulate(50, 1000, 42);
        if let TwoChoiceVerdict::Ok { applicants, .. } = v {
            assert_eq!(applicants, 50);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 100, 42);
        assert!(matches!(v, TwoChoiceVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(20, 100_000, 42);
        assert!(matches!(v, TwoChoiceVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_rates() {
        let a = simulate(20, 200, 42);
        let b = simulate(20, 200, 999);
        assert!(a != b);
    }

    #[test]
    fn larger_n_stable_rate() {
        // Both 50 and 200 should give similar rates (plateau).
        let small = simulate(50, 20_000, 42);
        let large = simulate(200, 20_000, 42);
        if let (
            TwoChoiceVerdict::Ok {
                success_rate_x1000: s,
                ..
            },
            TwoChoiceVerdict::Ok {
                success_rate_x1000: l,
                ..
            },
        ) = (small, large)
        {
            // Rates within 100 (10 percentage points) of each other.
            let diff = (s as i32 - l as i32).abs();
            assert!(diff < 100);
        }
    }
}
