//! # Monte-Carlo Simulated Annealing Optimum
//!
//! Sim simulated annealing on a 1D quadratic landscape `f(x) = (x-7)^2`.
//! Cooling schedule `T(t) = T0 / (1+t)`. Reports best x found and
//! best f(x).
//!
//! Demonstrates the **MC.91** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kirkpatrick, Gelatt, Vecchi, Science 220 (1983) §3
//!  (simulated annealing).
//!
//! Run with: cargo run --example mc_simulated_annealing_optimum
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AnnealVerdict {
    Ok { best_x: f64, best_value: f64 },
    InvalidConfig,
}

pub fn simulate(iterations: u32, initial_x: f64, initial_temp: f64, seed: u64) -> AnnealVerdict {
    if iterations == 0 || initial_temp <= 0.0 || !initial_x.is_finite() {
        return AnnealVerdict::InvalidConfig;
    }
    let f = |x: f64| (x - 7.0).powi(2);
    let mut current_x = initial_x;
    let mut current_v = f(current_x);
    let mut best_x = current_x;
    let mut best_v = current_v;
    let mut rng_state = seed | 1;
    for t in 0..iterations {
        let temp = initial_temp / (1.0 + f64::from(t));
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        // Step: ±0.5 uniformly.
        let step = (r - 0.5) * 1.0;
        let new_x = current_x + step;
        let new_v = f(new_x);
        let dv = new_v - current_v;
        let r2 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if dv < 0.0 || r2 < (-dv / temp).exp() {
            current_x = new_x;
            current_v = new_v;
            if current_v < best_v {
                best_v = current_v;
                best_x = current_x;
            }
        }
    }
    AnnealVerdict::Ok {
        best_x,
        best_value: best_v,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_simulated_annealing_optimum")?;

    println!("converges: {:?}", simulate(1000, 0.0, 5.0, 42));
    println!("invalid: {:?}", simulate(0, 0.0, 5.0, 42));
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
    fn converges_near_optimum() {
        // Optimum at x=7, f(x)=0.
        let v = simulate(5000, 0.0, 5.0, 42);
        if let AnnealVerdict::Ok { best_x, .. } = v {
            assert!((best_x - 7.0).abs() < 2.0);
        }
    }

    #[test]
    fn best_value_lower_than_initial() {
        let v = simulate(1000, 0.0, 5.0, 42);
        if let AnnealVerdict::Ok { best_value, .. } = v {
            // f(0) = 49; should converge to lower value.
            assert!(best_value < 49.0);
        }
    }

    #[test]
    fn invalid_zero_iterations() {
        assert_eq!(simulate(0, 0.0, 5.0, 42), AnnealVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_temp() {
        assert_eq!(simulate(100, 0.0, 0.0, 42), AnnealVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_nan_x() {
        assert_eq!(
            simulate(100, f64::NAN, 5.0, 42),
            AnnealVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 0.0, 5.0, 42);
        let b = simulate(500, 0.0, 5.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn best_value_nonneg() {
        let v = simulate(500, 0.0, 5.0, 42);
        if let AnnealVerdict::Ok { best_value, .. } = v {
            assert!(best_value >= 0.0);
        }
    }

    #[test]
    fn more_iterations_lower_value() {
        let short = simulate(100, 0.0, 5.0, 42);
        let long = simulate(5000, 0.0, 5.0, 42);
        if let (AnnealVerdict::Ok { best_value: s, .. }, AnnealVerdict::Ok { best_value: l, .. }) =
            (short, long)
        {
            assert!(l <= s);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(500, 0.0, 5.0, 42);
        if let AnnealVerdict::Ok { best_x, best_value } = v {
            assert!(best_x.is_finite());
            assert!(best_value.is_finite());
        }
    }

    #[test]
    fn close_initial_converges_faster() {
        let far = simulate(100, -10.0, 5.0, 42);
        let near = simulate(100, 6.5, 5.0, 42);
        if let (AnnealVerdict::Ok { best_value: f, .. }, AnnealVerdict::Ok { best_value: n, .. }) =
            (far, near)
        {
            // Near initial should reach lower value.
            assert!(n <= f);
        }
    }

    #[test]
    fn negative_initial_still_works() {
        let v = simulate(500, -50.0, 5.0, 42);
        assert!(matches!(v, AnnealVerdict::Ok { .. }));
    }
}
