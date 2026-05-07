//! # Monte-Carlo Circuit Breaker Recovery Time
//!
//! Sim a circuit breaker: opens on failure spike, half-opens after
//! cooldown, closes after N successful probes. Returns mean
//! cycles-to-recover and the proportion that recovered within the
//! window.
//!
//! Demonstrates the **MC.172** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nygard, Release It! (2007) circuit-breaker pattern;
//!  Hystrix retry policies (Netflix 2012).
//!
//! Run with: cargo run --example mc_circuit_recovery_time
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RecoveryVerdict {
    Ok {
        mean_recovery_cycles: u32,
        recovered_pct_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    success_prob_pct: u32,
    success_threshold: u32,
    max_cycles: u32,
    trials: u32,
    seed: u64,
) -> RecoveryVerdict {
    if !(1..=100).contains(&success_prob_pct)
        || success_threshold == 0
        || max_cycles < 5
        || trials < 100
    {
        return RecoveryVerdict::InvalidConfig;
    }
    let p = success_prob_pct as f64 / 100.0;
    let mut state = seed | 1;
    let mut total_cycles: u64 = 0;
    let mut recovered = 0u32;
    for _ in 0..trials {
        let mut consecutive = 0u32;
        let mut cycles = 0u32;
        for _ in 0..max_cycles {
            cycles += 1;
            let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
            if r < p {
                consecutive += 1;
                if consecutive >= success_threshold {
                    recovered += 1;
                    break;
                }
            } else {
                consecutive = 0;
            }
        }
        total_cycles += cycles as u64;
    }
    let mean = (total_cycles / trials as u64) as u32;
    let pct = (recovered as f64 / trials as f64 * 10000.0) as u32;
    RecoveryVerdict::Ok {
        mean_recovery_cycles: mean,
        recovered_pct_x100: pct,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_circuit_recovery_time")?;

    println!("80%, threshold-3: {:?}", simulate(80, 3, 50, 1000, 42));
    println!("invalid: {:?}", simulate(0, 3, 50, 1000, 42));
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
    fn invalid_zero_success_prob() {
        assert_eq!(simulate(0, 3, 50, 1000, 42), RecoveryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_threshold() {
        assert_eq!(
            simulate(80, 0, 50, 1000, 42),
            RecoveryVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_cycles() {
        assert_eq!(simulate(80, 3, 2, 1000, 42), RecoveryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(80, 3, 50, 50, 42), RecoveryVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(80, 3, 50, 500, 42);
        let b = simulate(80, 3, 50, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn high_success_prob_recovers() {
        let v = simulate(95, 3, 50, 1000, 42);
        if let RecoveryVerdict::Ok {
            recovered_pct_x100, ..
        } = v
        {
            // 95% success → essentially always recovers within max_cycles.
            assert!(recovered_pct_x100 > 9000);
        }
    }

    #[test]
    fn low_success_prob_struggles() {
        let v = simulate(10, 3, 50, 1000, 42);
        if let RecoveryVerdict::Ok {
            recovered_pct_x100, ..
        } = v
        {
            // 10% success → low recovery rate.
            assert!(recovered_pct_x100 < 5000);
        }
    }

    #[test]
    fn higher_threshold_more_cycles() {
        let low = simulate(80, 1, 50, 1000, 42);
        let high = simulate(80, 5, 50, 1000, 42);
        if let (
            RecoveryVerdict::Ok {
                mean_recovery_cycles: l,
                ..
            },
            RecoveryVerdict::Ok {
                mean_recovery_cycles: h,
                ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn mean_cycles_le_max() {
        let v = simulate(80, 3, 50, 1000, 42);
        if let RecoveryVerdict::Ok {
            mean_recovery_cycles,
            ..
        } = v
        {
            assert!(mean_recovery_cycles <= 50);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(1, 1, 5, 100, 42);
        assert!(matches!(v, RecoveryVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(80, 3, 50, 10_000, 42);
        assert!(matches!(v, RecoveryVerdict::Ok { .. }));
    }

    #[test]
    fn always_succeed_quick_recovery() {
        let v = simulate(100, 3, 50, 1000, 42);
        if let RecoveryVerdict::Ok {
            mean_recovery_cycles,
            recovered_pct_x100,
        } = v
        {
            assert_eq!(mean_recovery_cycles, 3);
            assert_eq!(recovered_pct_x100, 10_000);
        }
    }
}
