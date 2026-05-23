//! # Monte-Carlo Chaos Monkey Failure Sim
//!
//! Inject random instance failures over a window. Returns failure
//! count, max simultaneous failures, and mean time to recovery.
//!
//! Demonstrates the **MC.37** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Netflix Chaos Monkey resilience testing.
//!
//! Run with: cargo run --example mc_chaos_monkey_failures
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChaosVerdict {
    Ok {
        total_failures: u32,
        max_concurrent: u32,
        mean_recovery_steps: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    instances: u32,
    fail_prob_per_step: f64,
    recovery_steps: u32,
    duration_steps: u32,
    seed: u64,
) -> ChaosVerdict {
    if instances == 0
        || duration_steps == 0
        || recovery_steps == 0
        || !fail_prob_per_step.is_finite()
        || !(0.0..=1.0).contains(&fail_prob_per_step)
    {
        return ChaosVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut down: Vec<u32> = vec![0; instances as usize];
    let mut total_failures: u32 = 0;
    let mut max_concurrent: u32 = 0;
    let mut total_downtime: u64 = 0;
    for _ in 0..duration_steps {
        let mut concurrent: u32 = 0;
        for slot in &mut down {
            if *slot > 0 {
                *slot -= 1;
                concurrent += 1;
                total_downtime += 1;
            } else if unit(&mut rng_state) < fail_prob_per_step {
                *slot = recovery_steps;
                total_failures += 1;
                concurrent += 1;
                total_downtime += 1;
            }
        }
        if concurrent > max_concurrent {
            max_concurrent = concurrent;
        }
    }
    let mean_recovery_steps = if total_failures > 0 {
        total_downtime as f64 / f64::from(total_failures)
    } else {
        0.0
    };
    ChaosVerdict::Ok {
        total_failures,
        max_concurrent,
        mean_recovery_steps,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_chaos_monkey_failures")?;

    println!("calm: {:?}", simulate(10, 0.001, 5, 1000, 42));
    println!("chaotic: {:?}", simulate(10, 0.05, 5, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.01, 5, 1000, 42));
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
    fn calm_few_failures() {
        let v = simulate(10, 0.001, 5, 100, 42);
        if let ChaosVerdict::Ok { total_failures, .. } = v {
            assert!(total_failures < 10);
        }
    }

    #[test]
    fn chaotic_many_failures() {
        let v = simulate(10, 0.10, 5, 1000, 42);
        if let ChaosVerdict::Ok { total_failures, .. } = v {
            assert!(total_failures > 50);
        }
    }

    #[test]
    fn invalid_zero_instances() {
        assert_eq!(simulate(0, 0.01, 5, 100, 42), ChaosVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(simulate(10, 0.01, 5, 0, 42), ChaosVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_recovery() {
        assert_eq!(simulate(10, 0.01, 0, 100, 42), ChaosVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_prob() {
        assert_eq!(simulate(10, -0.1, 5, 100, 42), ChaosVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(10, f64::NAN, 5, 100, 42),
            ChaosVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 0.05, 5, 100, 42);
        let b = simulate(10, 0.05, 5, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_concurrent_bounded() {
        let v = simulate(10, 0.5, 5, 100, 42);
        if let ChaosVerdict::Ok { max_concurrent, .. } = v {
            assert!(max_concurrent <= 10);
        }
    }

    #[test]
    fn zero_prob_no_failures() {
        let v = simulate(10, 0.0, 5, 100, 42);
        if let ChaosVerdict::Ok { total_failures, .. } = v {
            assert_eq!(total_failures, 0);
        }
    }
}
