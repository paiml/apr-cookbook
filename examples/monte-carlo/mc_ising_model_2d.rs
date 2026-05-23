//! # Monte-Carlo 2D Ising Model
//!
//! Sim the 2D Ising model with Metropolis updates: spins flip with
//! probability min(1, exp(-ΔE/T)). Returns final magnetization
//! (×100) and energy per spin.
//!
//! Demonstrates the **MC.194** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Onsager, "Crystal statistics. I. A two-dimensional
//!  model with an order-disorder transition" Phys. Rev. 65 (1944);
//!  Tc/J ≈ 2.27 critical temperature.
//!
//! Run with: cargo run --example mc_ising_model_2d
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IsingVerdict {
    Ok {
        magnetization_x100: i32,
        energy_per_spin_x100: i32,
    },
    InvalidConfig,
}

pub fn simulate(grid_size: u32, temperature_x100: u32, sweeps: u32, seed: u64) -> IsingVerdict {
    if !(5..=50).contains(&grid_size) || temperature_x100 == 0 || sweeps < 10 {
        return IsingVerdict::InvalidConfig;
    }
    let n = grid_size as usize;
    let t = temperature_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut spins: Vec<i32> = (0..n * n)
        .map(|_| {
            if (lcg(&mut state) >> 32) & 1 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    for _ in 0..sweeps * (n * n) as u32 {
        let i = (lcg(&mut state) as usize) % n;
        let j = (lcg(&mut state) as usize) % n;
        let s = spins[i * n + j];
        let up = spins[((i + n - 1) % n) * n + j];
        let down = spins[((i + 1) % n) * n + j];
        let left = spins[i * n + (j + n - 1) % n];
        let right = spins[i * n + (j + 1) % n];
        let delta_e = 2 * s * (up + down + left + right);
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        if delta_e <= 0 || r < (-(delta_e as f64) / t).exp() {
            spins[i * n + j] = -s;
        }
    }
    let total: i32 = spins.iter().sum();
    let mag = total as f64 / (n * n) as f64;
    let mut energy: i32 = 0;
    for i in 0..n {
        for j in 0..n {
            let s = spins[i * n + j];
            let down = spins[((i + 1) % n) * n + j];
            let right = spins[i * n + (j + 1) % n];
            energy -= s * (down + right);
        }
    }
    let energy_per_spin = energy as f64 / (n * n) as f64;
    IsingVerdict::Ok {
        magnetization_x100: (mag.abs() * 100.0) as i32,
        energy_per_spin_x100: (energy_per_spin * 100.0) as i32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_ising_model_2d")?;

    println!("low T: {:?}", simulate(10, 100, 50, 42));
    println!("high T: {:?}", simulate(10, 500, 50, 42));
    println!("invalid: {:?}", simulate(2, 100, 10, 42));
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
    fn invalid_too_small_grid() {
        assert_eq!(simulate(2, 100, 10, 42), IsingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_temperature() {
        assert_eq!(simulate(10, 0, 10, 42), IsingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_sweeps() {
        assert_eq!(simulate(10, 100, 5, 42), IsingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(8, 200, 20, 42);
        let b = simulate(8, 200, 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn low_temp_high_magnetization() {
        // T < Tc → ordered phase, |M| close to 1.
        let v = simulate(10, 100, 100, 42);
        if let IsingVerdict::Ok {
            magnetization_x100, ..
        } = v
        {
            assert!(magnetization_x100 > 50);
        }
    }

    #[test]
    fn high_temp_lower_magnetization() {
        // T > Tc → disordered phase, |M| smaller than low-T.
        let low = simulate(10, 150, 100, 42);
        let high = simulate(10, 500, 100, 42);
        if let (
            IsingVerdict::Ok {
                magnetization_x100: l,
                ..
            },
            IsingVerdict::Ok {
                magnetization_x100: h,
                ..
            },
        ) = (low, high)
        {
            assert!(h <= l);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(5, 1, 10, 42);
        assert!(matches!(v, IsingVerdict::Ok { .. }));
    }

    #[test]
    fn many_sweeps_handled() {
        let v = simulate(10, 200, 200, 42);
        assert!(matches!(v, IsingVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(10, 200, 50, 42);
        let b = simulate(10, 200, 50, 999);
        assert!(a != b);
    }

    #[test]
    fn magnetization_in_zero_one() {
        let v = simulate(10, 200, 50, 42);
        if let IsingVerdict::Ok {
            magnetization_x100, ..
        } = v
        {
            assert!(magnetization_x100 <= 100);
        }
    }

    #[test]
    fn energy_finite() {
        let v = simulate(10, 200, 50, 42);
        if let IsingVerdict::Ok {
            energy_per_spin_x100,
            ..
        } = v
        {
            assert!(energy_per_spin_x100.abs() < 10_000);
        }
    }
}
