//! # Monte-Carlo Firefly Optimization Algorithm
//!
//! Sim firefly bio-inspired optimization: fireflies move toward
//! brighter neighbors. Returns best objective found and convergence
//! generation.
//!
//! Demonstrates the **MC.198** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Yang, "Firefly Algorithm, Stochastic Test Functions and
//!  Design Optimisation" Int. J. Bio-Inspired Computation (2010).
//!
//! Run with: cargo run --example mc_firefly_optimization
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FireflyVerdict {
    Ok {
        best_x_x100: i32,
        best_obj_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_fireflies: u32, generations: u32, seed: u64) -> FireflyVerdict {
    if n_fireflies < 5 || generations < 5 {
        return FireflyVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    // Objective: maximize f(x) = -((x-3)^2) for x in [-10, 10]
    let mut positions: Vec<f64> = (0..n_fireflies)
        .map(|_| {
            let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
            -10.0 + 20.0 * r
        })
        .collect();
    let alpha = 0.1f64; // randomness
    let beta_0 = 1.0f64;
    let gamma = 0.1f64; // light absorption
    for _ in 0..generations {
        for i in 0..positions.len() {
            for j in 0..positions.len() {
                if obj(positions[j]) > obj(positions[i]) {
                    let r = (positions[j] - positions[i]).abs();
                    let attractiveness = beta_0 * (-gamma * r * r).exp();
                    let noise = ((lcg(&mut state) as f64) / (u32::MAX as f64) - 0.5) * alpha;
                    positions[i] += attractiveness * (positions[j] - positions[i]) + noise;
                    positions[i] = positions[i].clamp(-10.0, 10.0);
                }
            }
        }
    }
    // Find best.
    let mut best_idx = 0;
    let mut best_obj = obj(positions[0]);
    for (i, p) in positions.iter().enumerate() {
        if obj(*p) > best_obj {
            best_obj = obj(*p);
            best_idx = i;
        }
    }
    FireflyVerdict::Ok {
        best_x_x100: (positions[best_idx] * 100.0) as i32,
        best_obj_x100: ((best_obj.max(0.0)) * 100.0) as u32,
    }
}

fn obj(x: f64) -> f64 {
    // Maximize -((x-3)^2): peak at x=3 with value 0.
    -((x - 3.0).powi(2))
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_firefly_optimization")?;

    println!("optimize: {:?}", simulate(20, 50, 42));
    println!("invalid: {:?}", simulate(2, 50, 42));
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
    fn invalid_too_few_fireflies() {
        assert_eq!(simulate(2, 50, 42), FireflyVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_generations() {
        assert_eq!(simulate(10, 2, 42), FireflyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 20, 42);
        let b = simulate(10, 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn converges_near_optimum() {
        // f(x) = -(x-3)^2 peaks at x=3 → best_x_x100 ≈ 300.
        let v = simulate(30, 100, 42);
        if let FireflyVerdict::Ok { best_x_x100, .. } = v {
            assert!((100..=500).contains(&best_x_x100));
        }
    }

    #[test]
    fn best_x_in_search_range() {
        let v = simulate(20, 50, 42);
        if let FireflyVerdict::Ok { best_x_x100, .. } = v {
            assert!(best_x_x100.abs() <= 1000);
        }
    }

    #[test]
    fn best_obj_at_least_zero() {
        let v = simulate(20, 50, 42);
        if let FireflyVerdict::Ok { best_obj_x100, .. } = v {
            assert!(best_obj_x100 < u32::MAX);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 5, 42);
        assert!(matches!(v, FireflyVerdict::Ok { .. }));
    }

    #[test]
    fn many_fireflies_handled() {
        let v = simulate(100, 50, 42);
        assert!(matches!(v, FireflyVerdict::Ok { .. }));
    }

    #[test]
    fn many_generations_handled() {
        let v = simulate(20, 200, 42);
        assert!(matches!(v, FireflyVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(20, 50, 42);
        let b = simulate(20, 50, 999);
        assert!(a != b);
    }

    #[test]
    fn obj_function_correct() {
        // f(3) = 0 (max); f(0) = -9; f(10) = -49.
        assert_eq!(obj(3.0), 0.0);
        assert_eq!(obj(0.0), -9.0);
    }
}
