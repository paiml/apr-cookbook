//! # Monte-Carlo Walking Meander
//!
//! Sim a pedestrian walking N steps, each step in a heading randomly
//! perturbed from previous by ±max_turn_deg. Reports total drift
//! (Euclidean from origin) and meander ratio (drift / N).
//!
//! Demonstrates the **MC.112** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: correlated random walk (Kareiva & Shigesada 1983);
//!  pedestrian dynamics in transportation engineering.
//!
//! Run with: cargo run --example mc_walking_meander
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MeanderVerdict {
    Ok {
        drift: f64,
        meander_ratio: f64,
        final_heading_deg: f64,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, max_turn_deg: f64, seed: u64) -> MeanderVerdict {
    if steps == 0 || !(0.0..=180.0).contains(&max_turn_deg) {
        return MeanderVerdict::InvalidConfig;
    }
    let mut x: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut heading_deg: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..steps {
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let turn = (r - 0.5) * 2.0 * max_turn_deg;
        heading_deg += turn;
        let rad = heading_deg.to_radians();
        x += rad.cos();
        y += rad.sin();
    }
    let drift = (x * x + y * y).sqrt();
    MeanderVerdict::Ok {
        drift,
        meander_ratio: drift / f64::from(steps),
        final_heading_deg: heading_deg.rem_euclid(360.0),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_walking_meander")?;

    println!("low turn: {:?}", simulate(1000, 5.0, 42));
    println!("high turn: {:?}", simulate(1000, 90.0, 42));
    println!("invalid: {:?}", simulate(0, 5.0, 42));
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
    fn straight_walk_high_drift() {
        // Zero turn → straight line → drift = steps.
        let v = simulate(100, 0.0, 42);
        if let MeanderVerdict::Ok { drift, .. } = v {
            assert!((drift - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn high_turn_low_drift() {
        let lo = simulate(1000, 5.0, 42);
        let hi = simulate(1000, 180.0, 42);
        if let (MeanderVerdict::Ok { drift: l, .. }, MeanderVerdict::Ok { drift: h, .. }) = (lo, hi)
        {
            assert!(l > h);
        }
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(0, 5.0, 42), MeanderVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_turn() {
        assert_eq!(simulate(100, -1.0, 42), MeanderVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_turn_above_180() {
        assert_eq!(simulate(100, 200.0, 42), MeanderVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 30.0, 42);
        let b = simulate(500, 30.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn drift_le_steps() {
        let v = simulate(1000, 30.0, 42);
        if let MeanderVerdict::Ok { drift, .. } = v {
            assert!(drift <= 1000.0);
        }
    }

    #[test]
    fn meander_ratio_in_unit_range() {
        let v = simulate(1000, 30.0, 42);
        if let MeanderVerdict::Ok { meander_ratio, .. } = v {
            assert!((0.0..=1.0).contains(&meander_ratio));
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 30.0, 42);
        if let MeanderVerdict::Ok {
            drift,
            meander_ratio,
            final_heading_deg,
        } = v
        {
            assert!(drift.is_finite());
            assert!(meander_ratio.is_finite());
            assert!(final_heading_deg.is_finite());
        }
    }

    #[test]
    fn final_heading_in_360_range() {
        let v = simulate(100, 30.0, 42);
        if let MeanderVerdict::Ok {
            final_heading_deg, ..
        } = v
        {
            assert!((0.0..360.0).contains(&final_heading_deg));
        }
    }

    #[test]
    fn single_step_drift_one() {
        let v = simulate(1, 0.0, 42);
        if let MeanderVerdict::Ok { drift, .. } = v {
            assert!((drift - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(10_000, 30.0, 42);
        assert!(matches!(v, MeanderVerdict::Ok { .. }));
    }
}
