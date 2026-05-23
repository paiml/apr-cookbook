//! # Monte-Carlo Brownian Bridge Path
//!
//! Sim a Brownian bridge: a Wiener process pinned at both endpoints
//! (B(0)=a, B(T)=b). Returns max excursion from the linear
//! interpolation between endpoints.
//!
//! Demonstrates the **MC.182** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lévy 1948 construction; Karatzas & Shreve, Brownian
//!  Motion and Stochastic Calculus §5.6.B.
//!
//! Run with: cargo run --example mc_brownian_bridge_path
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BridgeVerdict {
    Ok {
        max_excursion_x100: u32,
        endpoint_drift_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, end_offset_x100: i32, seed: u64) -> BridgeVerdict {
    if steps < 10 {
        return BridgeVerdict::InvalidConfig;
    }
    let end = end_offset_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let dt = 1.0 / steps as f64;
    let mut x = 0.0f64;
    let mut max_exc = 0.0f64;
    let mut path: Vec<f64> = vec![0.0; (steps + 1) as usize];
    // Generate raw Wiener path
    for p in path.iter_mut().skip(1) {
        let u1 = ((lcg(&mut state) as f64) / (u32::MAX as f64)).max(1e-10);
        let u2 = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x += z * dt.sqrt();
        *p = x;
    }
    let raw_end = path[steps as usize];
    // Bridge transform: B(t) = W(t) - (t/T) * (W(T) - desired_end)
    let total = steps as f64;
    for (i, p) in path.iter_mut().enumerate() {
        let t = i as f64 / total;
        *p -= t * (raw_end - end);
        let lin = end * t;
        let exc = (*p - lin).abs();
        if exc > max_exc {
            max_exc = exc;
        }
    }
    let drift = (path[steps as usize] - end).abs();
    BridgeVerdict::Ok {
        max_excursion_x100: (max_exc * 100.0) as u32,
        endpoint_drift_x100: (drift * 100.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_brownian_bridge_path")?;

    println!("end=0: {:?}", simulate(1000, 0, 42));
    println!("end=2: {:?}", simulate(1000, 200, 42));
    println!("invalid: {:?}", simulate(5, 0, 42));
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
    fn invalid_too_few_steps() {
        assert_eq!(simulate(5, 0, 42), BridgeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 0, 42);
        let b = simulate(100, 0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn endpoint_pinned() {
        // Bridge transform pins the endpoint exactly.
        let v = simulate(100, 0, 42);
        if let BridgeVerdict::Ok {
            endpoint_drift_x100,
            ..
        } = v
        {
            // Drift should be at most rounding noise.
            assert!(endpoint_drift_x100 <= 1);
        }
    }

    #[test]
    fn endpoint_pinned_nonzero() {
        let v = simulate(100, 200, 42);
        if let BridgeVerdict::Ok {
            endpoint_drift_x100,
            ..
        } = v
        {
            assert!(endpoint_drift_x100 <= 1);
        }
    }

    #[test]
    fn max_excursion_finite() {
        let v = simulate(100, 0, 42);
        if let BridgeVerdict::Ok {
            max_excursion_x100, ..
        } = v
        {
            assert!(max_excursion_x100 < u32::MAX);
        }
    }

    #[test]
    fn longer_walk_handled() {
        let v = simulate(10_000, 0, 42);
        assert!(matches!(v, BridgeVerdict::Ok { .. }));
    }

    #[test]
    fn min_steps_accepted() {
        let v = simulate(10, 0, 42);
        assert!(matches!(v, BridgeVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 0, 42);
        let b = simulate(100, 0, 999);
        assert!(a != b);
    }

    #[test]
    fn negative_end_offset_handled() {
        let v = simulate(100, -200, 42);
        if let BridgeVerdict::Ok {
            endpoint_drift_x100,
            ..
        } = v
        {
            assert!(endpoint_drift_x100 <= 1);
        }
    }

    #[test]
    fn many_walks_consistent() {
        // Across multiple seeds, endpoint pinning should always hold.
        for s in 0..20u64 {
            let v = simulate(100, 0, s);
            if let BridgeVerdict::Ok {
                endpoint_drift_x100,
                ..
            } = v
            {
                assert!(endpoint_drift_x100 <= 1);
            }
        }
    }

    #[test]
    fn excursion_at_least_zero() {
        let v = simulate(100, 0, 42);
        if let BridgeVerdict::Ok {
            max_excursion_x100, ..
        } = v
        {
            assert!(max_excursion_x100 < u32::MAX);
        }
    }

    #[test]
    fn high_step_count_handled() {
        let v = simulate(100_000, 100, 42);
        assert!(matches!(v, BridgeVerdict::Ok { .. }));
    }
}
