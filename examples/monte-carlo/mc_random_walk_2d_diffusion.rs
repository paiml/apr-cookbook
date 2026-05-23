//! # Monte-Carlo 2D Random Walk Diffusion
//!
//! Sim N independent 2D random walks of `steps` steps each.
//! Reports mean squared displacement (should scale linearly with
//! steps per Einstein's diffusion law).
//!
//! Demonstrates the **MC.77** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Einstein, On the Movement of Small Particles… (Annalen
//!  der Physik, 1905); Pearson, Nature 72 (1905).
//!
//! Run with: cargo run --example mc_random_walk_2d_diffusion
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiffusionVerdict {
    Ok {
        mean_squared_displacement: f64,
        max_displacement: f64,
    },
    InvalidConfig,
}

pub fn simulate(walks: u32, steps: u32, seed: u64) -> DiffusionVerdict {
    if walks == 0 || steps == 0 {
        return DiffusionVerdict::InvalidConfig;
    }
    let mut total_sqd: f64 = 0.0;
    let mut max_disp: f64 = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..walks {
        let mut x: i32 = 0;
        let mut y: i32 = 0;
        for _ in 0..steps {
            let dir = (lcg(&mut rng_state) >> 32) as u32 % 4;
            match dir {
                0 => x += 1,
                1 => x -= 1,
                2 => y += 1,
                _ => y -= 1,
            }
        }
        let sqd = (x * x + y * y) as f64;
        total_sqd += sqd;
        let disp = sqd.sqrt();
        if disp > max_disp {
            max_disp = disp;
        }
    }
    DiffusionVerdict::Ok {
        mean_squared_displacement: total_sqd / f64::from(walks),
        max_displacement: max_disp,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_walk_2d_diffusion")?;

    println!("100 steps: {:?}", simulate(1000, 100, 42));
    println!("1000 steps: {:?}", simulate(1000, 1000, 42));
    println!("invalid: {:?}", simulate(0, 100, 42));
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
    fn msd_scales_with_steps() {
        // Einstein's law: MSD ≈ steps for 2D unit walk.
        let v100 = simulate(2000, 100, 42);
        let v1000 = simulate(2000, 1000, 42);
        if let (
            DiffusionVerdict::Ok {
                mean_squared_displacement: m100,
                ..
            },
            DiffusionVerdict::Ok {
                mean_squared_displacement: m1000,
                ..
            },
        ) = (v100, v1000)
        {
            // 1000 / 100 = 10×, allow wide bounds for 2k samples.
            let ratio = m1000 / m100;
            assert!(ratio > 5.0 && ratio < 20.0);
        }
    }

    #[test]
    fn msd_around_steps_for_2d() {
        // MSD for unit 2D walk ≈ 1.0 × steps (variance per step = 1).
        let v = simulate(5000, 100, 42);
        if let DiffusionVerdict::Ok {
            mean_squared_displacement,
            ..
        } = v
        {
            // Allow factor-2 tolerance for 5k samples.
            assert!(mean_squared_displacement > 50.0);
            assert!(mean_squared_displacement < 200.0);
        }
    }

    #[test]
    fn invalid_zero_walks() {
        assert_eq!(simulate(0, 100, 42), DiffusionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(100, 0, 42), DiffusionVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_le_steps() {
        // Max possible displacement = steps (all moves in same direction).
        let v = simulate(100, 50, 42);
        if let DiffusionVerdict::Ok {
            max_displacement, ..
        } = v
        {
            assert!(max_displacement <= 50.0);
        }
    }

    #[test]
    fn max_geq_zero() {
        let v = simulate(100, 50, 42);
        if let DiffusionVerdict::Ok {
            max_displacement, ..
        } = v
        {
            assert!(max_displacement >= 0.0);
        }
    }

    #[test]
    fn single_walk_works() {
        let v = simulate(1, 100, 42);
        if let DiffusionVerdict::Ok {
            mean_squared_displacement,
            ..
        } = v
        {
            assert!(mean_squared_displacement >= 0.0);
        }
    }

    #[test]
    fn one_step_msd_one() {
        let v = simulate(10_000, 1, 42);
        if let DiffusionVerdict::Ok {
            mean_squared_displacement,
            ..
        } = v
        {
            // After 1 step, displacement² = 1 always.
            assert!((mean_squared_displacement - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn msd_nonneg() {
        let v = simulate(100, 100, 42);
        if let DiffusionVerdict::Ok {
            mean_squared_displacement,
            ..
        } = v
        {
            assert!(mean_squared_displacement >= 0.0);
        }
    }

    #[test]
    fn zero_steps_in_walks_rejected_separately() {
        // Already tested: simulate(N, 0, _) → InvalidConfig.
        assert_eq!(simulate(100, 0, 42), DiffusionVerdict::InvalidConfig);
    }
}
