//! # Monte-Carlo Inhomogeneous Poisson via Thinning
//!
//! Sim an inhomogeneous Poisson process with rate λ(t) = λ_max·sin²(t)
//! using Lewis-Shedler thinning: propose at rate λ_max, accept with
//! probability λ(t)/λ_max. Returns event count and acceptance rate.
//!
//! Demonstrates the **MC.183** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lewis & Shedler, "Simulation of nonhomogeneous Poisson
//!  processes by thinning" Naval Research Logistics 26(3) (1979).
//!
//! Run with: cargo run --example mc_inhomog_poisson_thinning
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThinningVerdict {
    Ok {
        event_count: u32,
        accept_rate_x1000: u32,
    },
    InvalidConfig,
}

pub fn simulate(duration_x100: u32, lambda_max_x100: u32, seed: u64) -> ThinningVerdict {
    if duration_x100 < 100 || lambda_max_x100 == 0 {
        return ThinningVerdict::InvalidConfig;
    }
    let duration = duration_x100 as f64 / 100.0;
    let lambda_max = lambda_max_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut t = 0.0f64;
    let mut events = 0u32;
    let mut proposals = 0u32;
    while t < duration {
        // Inter-arrival from Exp(lambda_max)
        let u = ((lcg(&mut state) as f64) / (u32::MAX as f64)).max(1e-10);
        let dt = -(1.0 - u).ln() / lambda_max;
        t += dt;
        if t >= duration {
            break;
        }
        proposals += 1;
        // Acceptance: lambda(t) / lambda_max where lambda(t) = lambda_max * sin²(t)
        let lambda_t = lambda_max * t.sin().powi(2);
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        if r < lambda_t / lambda_max {
            events += 1;
        }
    }
    let rate = if proposals > 0 {
        (events as f64 / proposals as f64 * 1000.0) as u32
    } else {
        0
    };
    ThinningVerdict::Ok {
        event_count: events,
        accept_rate_x1000: rate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inhomog_poisson_thinning")?;

    println!("T=10, λ=2: {:?}", simulate(1000, 200, 42));
    println!("invalid: {:?}", simulate(50, 200, 42));
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
        assert_eq!(simulate(50, 200, 42), ThinningVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_lambda() {
        assert_eq!(simulate(1000, 0, 42), ThinningVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 200, 42);
        let b = simulate(1000, 200, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_lambda_more_events() {
        let low = simulate(1000, 100, 42);
        let high = simulate(1000, 500, 42);
        if let (
            ThinningVerdict::Ok { event_count: l, .. },
            ThinningVerdict::Ok { event_count: h, .. },
        ) = (low, high)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn accept_rate_in_zero_one() {
        let v = simulate(1000, 200, 42);
        if let ThinningVerdict::Ok {
            accept_rate_x1000, ..
        } = v
        {
            assert!(accept_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn accept_rate_near_half_for_sin_squared() {
        // Mean of sin² over a full period = 1/2 → accept rate ~ 0.5.
        let v = simulate(2000, 100, 42);
        if let ThinningVerdict::Ok {
            accept_rate_x1000, ..
        } = v
        {
            assert!((350..=650).contains(&accept_rate_x1000));
        }
    }

    #[test]
    fn longer_duration_more_events() {
        let short = simulate(500, 200, 42);
        let long = simulate(2000, 200, 42);
        if let (
            ThinningVerdict::Ok { event_count: s, .. },
            ThinningVerdict::Ok { event_count: l, .. },
        ) = (short, long)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(100, 1, 42);
        assert!(matches!(v, ThinningVerdict::Ok { .. }));
    }

    #[test]
    fn many_events_handled() {
        let v = simulate(10_000, 1000, 42);
        assert!(matches!(v, ThinningVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(1000, 200, 42);
        let b = simulate(1000, 200, 999);
        assert!(a != b);
    }
}
