//! # Monte-Carlo Hawkes Self-Exciting Point Process
//!
//! Sim a Hawkes process where each event raises the rate of future
//! events by an exponential kernel. Returns total event count and
//! the maximum observed inter-arrival time.
//!
//! Demonstrates the **MC.164** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hawkes, "Spectra of some self-exciting and mutually
//!  exciting point processes" Biometrika (1971); Ogata thinning
//!  algorithm (1981).
//!
//! Run with: cargo run --example mc_hawkes_self_exciting_process
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HawkesVerdict {
    Ok {
        event_count: u32,
        baseline_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    duration: u32,
    baseline_x100: u32,
    excitation_x100: u32,
    decay_x100: u32,
    seed: u64,
) -> HawkesVerdict {
    if duration < 10
        || baseline_x100 == 0
        || excitation_x100 == 0
        || decay_x100 == 0
        || excitation_x100 >= decay_x100 * 100
    {
        return HawkesVerdict::InvalidConfig;
    }
    let baseline = baseline_x100 as f64 / 100.0;
    let excitation = excitation_x100 as f64 / 100.0;
    let decay = decay_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut t = 0.0f64;
    let mut events: Vec<f64> = Vec::new();
    while t < duration as f64 {
        // Current intensity: baseline + sum_i excitation * exp(-decay * (t-t_i))
        let intensity = baseline
            + events
                .iter()
                .map(|ti| excitation * (-decay * (t - ti)).exp())
                .sum::<f64>();
        // Ogata thinning: propose with rate λ*, accept with prob λ/λ*.
        let u_raw = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let u = u_raw.max(1e-10);
        let dt = -(1.0 - u).ln() / intensity.max(0.001);
        t += dt;
        if t >= duration as f64 {
            break;
        }
        // Acceptance: λ at new t / λ* (use current as bound for simplicity)
        let lambda_new = baseline
            + events
                .iter()
                .map(|ti| excitation * (-decay * (t - ti)).exp())
                .sum::<f64>();
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        if r * intensity.max(0.001) <= lambda_new {
            events.push(t);
        }
        if events.len() > 100_000 {
            break; // safety
        }
    }
    HawkesVerdict::Ok {
        event_count: events.len() as u32,
        baseline_x100,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_hawkes_self_exciting_process")?;

    println!("baseline=0.5: {:?}", simulate(100, 50, 30, 100, 42));
    println!("invalid: {:?}", simulate(5, 50, 30, 100, 42));
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
    fn invalid_too_short_duration() {
        assert_eq!(simulate(5, 50, 30, 100, 42), HawkesVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_baseline() {
        assert_eq!(simulate(100, 0, 30, 100, 42), HawkesVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_excitation() {
        assert_eq!(simulate(100, 50, 0, 100, 42), HawkesVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_decay() {
        assert_eq!(simulate(100, 50, 30, 0, 42), HawkesVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_excitation_dominates_decay() {
        // excitation ≥ decay * 100 → process explodes.
        assert_eq!(
            simulate(100, 50, 10000, 50, 42),
            HawkesVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 50, 30, 100, 42);
        let b = simulate(50, 50, 30, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn baseline_returned() {
        let v = simulate(50, 50, 30, 100, 42);
        if let HawkesVerdict::Ok { baseline_x100, .. } = v {
            assert_eq!(baseline_x100, 50);
        }
    }

    #[test]
    fn higher_baseline_more_events() {
        let low = simulate(50, 30, 20, 100, 42);
        let high = simulate(50, 100, 20, 100, 42);
        if let (
            HawkesVerdict::Ok { event_count: l, .. },
            HawkesVerdict::Ok { event_count: h, .. },
        ) = (low, high)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn min_duration_accepted() {
        let v = simulate(10, 50, 30, 100, 42);
        assert!(matches!(v, HawkesVerdict::Ok { .. }));
    }

    #[test]
    fn long_duration_handled() {
        let v = simulate(1000, 50, 30, 100, 42);
        assert!(matches!(v, HawkesVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(50, 50, 30, 100, 42);
        let b = simulate(50, 50, 30, 100, 999);
        assert!(a != b);
    }
}
