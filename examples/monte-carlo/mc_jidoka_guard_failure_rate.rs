//! # Monte-Carlo Jidoka Guard Failure-Rate
//!
//! Sim a simulation pipeline guarded by jidoka invariants (cf.
//! `simular`'s `JidokaGuard`): each step has an independent failure
//! probability; pipeline halts on first violation. Returns mean
//! steps-to-violation.
//!
//! Demonstrates the **MC.168** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Toyota Production System jidoka principle (Ohno 1988);
//!  simular crate `JidokaGuard` invariant-checking pattern (now
//!  ../aprender/crates/aprender-simulate).
//!
//! Run with: cargo run --example mc_jidoka_guard_failure_rate
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum JidokaVerdict {
    Ok {
        mean_steps_to_violation: u32,
        violation_rate_x1000: u32,
    },
    InvalidConfig,
}

pub fn simulate(failure_prob_x1000: u32, max_steps: u32, trials: u32, seed: u64) -> JidokaVerdict {
    if !(1..=999).contains(&failure_prob_x1000) || max_steps < 10 || trials < 100 {
        return JidokaVerdict::InvalidConfig;
    }
    let p = failure_prob_x1000 as f64 / 1000.0;
    let mut state = seed | 1;
    let mut total_steps: u64 = 0;
    let mut violations = 0u32;
    for _ in 0..trials {
        let mut steps = 0u32;
        for _ in 0..max_steps {
            steps += 1;
            let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
            if r < p {
                violations += 1;
                break;
            }
        }
        total_steps += steps as u64;
    }
    let mean = (total_steps / trials as u64) as u32;
    let rate = (violations as f64 / trials as f64 * 1000.0) as u32;
    JidokaVerdict::Ok {
        mean_steps_to_violation: mean,
        violation_rate_x1000: rate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_jidoka_guard_failure_rate")?;

    println!("p=0.05: {:?}", simulate(50, 100, 1000, 42));
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
    fn invalid_zero_failure_prob() {
        assert_eq!(simulate(0, 100, 1000, 42), JidokaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_failure_prob() {
        assert_eq!(simulate(1000, 100, 1000, 42), JidokaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_steps() {
        assert_eq!(simulate(50, 5, 1000, 42), JidokaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(50, 100, 50, 42), JidokaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 100, 500, 42);
        let c = simulate(50, 100, 500, 42);
        assert_eq!(a, c);
    }

    #[test]
    fn lower_failure_prob_more_steps() {
        let high_p = simulate(200, 100, 1000, 42);
        let low_p = simulate(20, 100, 1000, 42);
        if let (
            JidokaVerdict::Ok {
                mean_steps_to_violation: h,
                ..
            },
            JidokaVerdict::Ok {
                mean_steps_to_violation: l,
                ..
            },
        ) = (high_p, low_p)
        {
            assert!(l > h);
        }
    }

    #[test]
    fn mean_steps_at_least_one() {
        let v = simulate(50, 100, 1000, 42);
        if let JidokaVerdict::Ok {
            mean_steps_to_violation,
            ..
        } = v
        {
            assert!(mean_steps_to_violation >= 1);
        }
    }

    #[test]
    fn rate_in_zero_one() {
        let v = simulate(50, 100, 1000, 42);
        if let JidokaVerdict::Ok {
            violation_rate_x1000,
            ..
        } = v
        {
            assert!(violation_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn high_failure_high_violation_rate() {
        let v = simulate(500, 100, 1000, 42);
        if let JidokaVerdict::Ok {
            violation_rate_x1000,
            ..
        } = v
        {
            // p=0.5 over 100 steps → essentially always violates.
            assert!(violation_rate_x1000 > 950);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(1, 10, 100, 42);
        assert!(matches!(v, JidokaVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(50, 100, 10_000, 42);
        assert!(matches!(v, JidokaVerdict::Ok { .. }));
    }

    #[test]
    fn mean_steps_le_max() {
        let v = simulate(50, 100, 500, 42);
        if let JidokaVerdict::Ok {
            mean_steps_to_violation,
            ..
        } = v
        {
            assert!(mean_steps_to_violation <= 100);
        }
    }
}
