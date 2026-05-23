//! # Monte-Carlo Brownian Motion Path
//!
//! Sim a discrete-time random walk approximating standard Brownian
//! motion (Wiener process). Each step is ±dt^0.5. Returns final
//! position and max-magnitude excursion.
//!
//! Demonstrates the **MC.139** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Brown 1827; Einstein 1905; Wiener process formal
//!  construction (1923).
//!
//! Run with: cargo run --example mc_brownian_motion_path
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BmVerdict {
    Ok {
        final_x1000: i32,
        max_excursion_x1000: u32,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, dt_x1000: u32, seed: u64) -> BmVerdict {
    if steps == 0 || dt_x1000 == 0 {
        return BmVerdict::InvalidConfig;
    }
    let dt = dt_x1000 as f64 / 1000.0;
    let step_size = dt.sqrt();
    let mut state = seed | 1;
    let mut x = 0.0f64;
    let mut max_excursion = 0.0f64;
    for _ in 0..steps {
        if (lcg(&mut state) >> 32) % 2 == 0 {
            x += step_size;
        } else {
            x -= step_size;
        }
        if x.abs() > max_excursion {
            max_excursion = x.abs();
        }
    }
    BmVerdict::Ok {
        final_x1000: (x * 1000.0) as i32,
        max_excursion_x1000: (max_excursion * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_brownian_motion_path")?;

    println!("walk-1000: {:?}", simulate(1000, 1, 42));
    println!("invalid: {:?}", simulate(0, 1, 42));
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
    fn invalid_zero_steps() {
        assert_eq!(simulate(0, 1, 42), BmVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_dt() {
        assert_eq!(simulate(100, 0, 42), BmVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 1, 42);
        let b = simulate(100, 1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_excursion_non_negative() {
        let v = simulate(100, 1, 42);
        if let BmVerdict::Ok {
            max_excursion_x1000,
            ..
        } = v
        {
            // u32 is naturally non-negative; sanity is finite.
            assert!(max_excursion_x1000 < u32::MAX);
        }
    }

    #[test]
    fn max_excursion_bounded_by_path_length() {
        // |max excursion| ≤ N * sqrt(dt). For N=100, dt=0.001,
        // bound = 100 * sqrt(0.001) ≈ 3.162 → 3162.
        let v = simulate(100, 1, 42);
        if let BmVerdict::Ok {
            max_excursion_x1000,
            ..
        } = v
        {
            assert!(max_excursion_x1000 <= 3163);
        }
    }

    #[test]
    fn final_within_excursion_bound() {
        let v = simulate(100, 1, 42);
        if let BmVerdict::Ok {
            final_x1000,
            max_excursion_x1000,
        } = v
        {
            assert!(final_x1000.unsigned_abs() <= max_excursion_x1000);
        }
    }

    #[test]
    fn longer_walk_larger_or_equal_excursion() {
        let short = simulate(100, 1, 42);
        let long = simulate(1000, 1, 42);
        if let (
            BmVerdict::Ok {
                max_excursion_x1000: s,
                ..
            },
            BmVerdict::Ok {
                max_excursion_x1000: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn larger_dt_larger_steps() {
        let small_dt = simulate(100, 1, 42);
        let large_dt = simulate(100, 100, 42);
        if let (
            BmVerdict::Ok {
                max_excursion_x1000: s,
                ..
            },
            BmVerdict::Ok {
                max_excursion_x1000: l,
                ..
            },
        ) = (small_dt, large_dt)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn variance_grows_with_n() {
        // E[X_N^2] = N*dt. Average over many seeds.
        let mut sum_sq: u64 = 0;
        for s in 0..100 {
            if let BmVerdict::Ok { final_x1000, .. } = simulate(100, 10, s as u64) {
                let f = final_x1000 as f64 / 1000.0;
                sum_sq += (f * f * 1000.0) as u64;
            }
        }
        // E[X^2] ≈ N*dt = 100*0.01 = 1.0; over 100 trials ≈ 100,000 (×1000).
        // Wide tolerance for finite-sample variance.
        assert!(sum_sq > 30_000 && sum_sq < 200_000);
    }

    #[test]
    fn small_walk_handled() {
        let v = simulate(1, 1, 42);
        assert!(matches!(v, BmVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(10_000, 1, 42);
        assert!(matches!(v, BmVerdict::Ok { .. }));
    }
}
