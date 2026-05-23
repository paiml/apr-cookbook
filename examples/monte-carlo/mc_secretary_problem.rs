//! # Monte-Carlo Secretary Problem
//!
//! Sim the optimal-stopping "best applicant" problem: of n applicants
//! shown sequentially, reject the first n/e (~37%), then pick the
//! first one better than all seen. Returns success rate.
//!
//! Demonstrates the **MC.151** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lindley, "Dynamic Programming and Decision Theory" (1961);
//!  Gilbert & Mosteller, "Recognizing the maximum of a sequence" JASA
//!  (1966).
//!
//! Run with: cargo run --example mc_secretary_problem
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SecretaryVerdict {
    Ok {
        success_rate_x1000: u32,
        applicants: u32,
    },
    InvalidConfig,
}

pub fn simulate(applicants: u32, trials: u32, seed: u64) -> SecretaryVerdict {
    if applicants < 5 || trials < 100 {
        return SecretaryVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let observe = (applicants as f64 / std::f64::consts::E) as usize;
    let mut successes = 0u32;
    for _ in 0..trials {
        // Rank permutation 1..=n.
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
        if chosen == Some(applicants) {
            successes += 1;
        }
    }
    let rate = (successes as f64 / trials as f64) * 1000.0;
    SecretaryVerdict::Ok {
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
    let _ctx = RecipeContext::new("mc_secretary_problem")?;

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
        assert_eq!(simulate(2, 100, 42), SecretaryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(10, 50, 42), SecretaryVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 1000, 42);
        let b = simulate(20, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn success_rate_near_1_over_e() {
        // Theoretical limit: 1/e ≈ 0.368 → 368.
        let v = simulate(100, 50_000, 42);
        if let SecretaryVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!((300..=420).contains(&success_rate_x1000));
        }
    }

    #[test]
    fn success_rate_in_zero_one() {
        let v = simulate(20, 1000, 42);
        if let SecretaryVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!(success_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn applicants_returned() {
        let v = simulate(50, 1000, 42);
        if let SecretaryVerdict::Ok { applicants, .. } = v {
            assert_eq!(applicants, 50);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 100, 42);
        assert!(matches!(v, SecretaryVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(20, 100_000, 42);
        assert!(matches!(v, SecretaryVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_rates() {
        let a = simulate(20, 200, 42);
        let b = simulate(20, 200, 999);
        assert!(a != b);
    }

    #[test]
    fn larger_n_converges_to_1_over_e() {
        // Theoretical success rate → 1/e as n → ∞. n=20 should still
        // be near 1/e but with more variance than n=100.
        let v = simulate(200, 20_000, 42);
        if let SecretaryVerdict::Ok {
            success_rate_x1000, ..
        } = v
        {
            assert!((300..=420).contains(&success_rate_x1000));
        }
    }
}
