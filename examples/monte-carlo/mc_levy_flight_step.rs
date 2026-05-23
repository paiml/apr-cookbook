//! # Monte-Carlo Lévy Flight Random Walk
//!
//! Sim a 1-D Lévy flight: each step is drawn from a heavy-tailed
//! Pareto-like distribution. Returns final position and the maximum
//! single-step magnitude observed (heavy tails should produce
//! occasional very large jumps).
//!
//! Demonstrates the **MC.163** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mandelbrot (1963) Lévy stable distributions; foraging
//!  models (Viswanathan 1996, albatross flight patterns).
//!
//! Run with: cargo run --example mc_levy_flight_step
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LevyVerdict {
    Ok {
        final_position_abs: u64,
        max_step_abs: u32,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, alpha_x100: u32, seed: u64) -> LevyVerdict {
    if steps < 100 || !(101..=300).contains(&alpha_x100) {
        return LevyVerdict::InvalidConfig;
    }
    let alpha = alpha_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut pos: i64 = 0;
    let mut max_step = 0u32;
    for _ in 0..steps {
        // Pareto-distributed magnitude: |x| = u^(-1/(α-1)) for u ~ Uniform(0,1).
        let u_raw = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let u = u_raw.max(1e-10);
        let mag = u.powf(-1.0 / (alpha - 1.0));
        // Direction
        let sign = if (lcg(&mut state) >> 32) % 2 == 0 {
            1i64
        } else {
            -1
        };
        let step = (mag.min(1e9) as i64) * sign;
        pos = pos.saturating_add(step);
        let abs_step = step.unsigned_abs() as u32;
        if abs_step > max_step {
            max_step = abs_step;
        }
    }
    LevyVerdict::Ok {
        final_position_abs: pos.unsigned_abs(),
        max_step_abs: max_step,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_levy_flight_step")?;

    println!("alpha=1.5: {:?}", simulate(1000, 150, 42));
    println!("alpha=2.5: {:?}", simulate(1000, 250, 42));
    println!("invalid: {:?}", simulate(50, 150, 42));
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
        assert_eq!(simulate(50, 150, 42), LevyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_alpha_too_low() {
        assert_eq!(simulate(1000, 100, 42), LevyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_alpha_too_high() {
        assert_eq!(simulate(1000, 301, 42), LevyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 150, 42);
        let b = simulate(500, 150, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_step_at_least_one() {
        let v = simulate(500, 150, 42);
        if let LevyVerdict::Ok { max_step_abs, .. } = v {
            assert!(max_step_abs >= 1);
        }
    }

    #[test]
    fn smaller_alpha_heavier_tails() {
        // alpha=1.5 (heavy) vs alpha=2.8 (lighter) → max step generally larger for 1.5.
        let heavy = simulate(2000, 150, 42);
        let light = simulate(2000, 280, 42);
        if let (
            LevyVerdict::Ok {
                max_step_abs: h, ..
            },
            LevyVerdict::Ok {
                max_step_abs: l, ..
            },
        ) = (heavy, light)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn min_steps_accepted() {
        let v = simulate(100, 150, 42);
        assert!(matches!(v, LevyVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(10_000, 150, 42);
        assert!(matches!(v, LevyVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(500, 150, 42);
        let b = simulate(500, 150, 999);
        assert!(a != b);
    }

    #[test]
    fn finite_position() {
        let v = simulate(500, 150, 42);
        if let LevyVerdict::Ok {
            final_position_abs, ..
        } = v
        {
            assert!(final_position_abs < u64::MAX);
        }
    }

    #[test]
    fn final_pos_le_n_times_max_step() {
        let v = simulate(500, 200, 42);
        if let LevyVerdict::Ok {
            final_position_abs,
            max_step_abs,
        } = v
        {
            // Trivially: |sum| ≤ n · max_step.
            assert!(final_position_abs <= 500u64 * max_step_abs as u64);
        }
    }
}
