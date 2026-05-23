//! # Monte-Carlo Pólya Urn Color Balance
//!
//! Sim Pólya's urn: start with `r` red and `b` black balls; on each
//! draw, return the ball plus add one of the same color. Returns
//! final composition fractions (×1000).
//!
//! Demonstrates the **MC.146** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pólya & Eggenberger, "Über die Statistik verketteter
//!  Vorgänge" (1923); reinforcement-process foundation.
//!
//! Run with: cargo run --example mc_polya_urn_color_balance
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PolyaVerdict {
    Ok {
        red_fraction_x1000: u32,
        total_balls: u32,
    },
    InvalidConfig,
}

pub fn simulate(initial_red: u32, initial_black: u32, draws: u32, seed: u64) -> PolyaVerdict {
    if initial_red == 0 && initial_black == 0 {
        return PolyaVerdict::InvalidConfig;
    }
    if draws == 0 {
        return PolyaVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut red = initial_red;
    let mut black = initial_black;
    for _ in 0..draws {
        let total = red + black;
        let r = lcg(&mut state) % total as u64;
        if r < red as u64 {
            red += 1;
        } else {
            black += 1;
        }
    }
    let total = red + black;
    PolyaVerdict::Ok {
        red_fraction_x1000: ((red as f64 / total as f64) * 1000.0) as u32,
        total_balls: total,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_polya_urn_color_balance")?;

    println!("balanced: {:?}", simulate(1, 1, 100, 42));
    println!("biased: {:?}", simulate(10, 1, 100, 42));
    println!("invalid: {:?}", simulate(0, 0, 100, 42));
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
    fn invalid_zero_balls() {
        assert_eq!(simulate(0, 0, 100, 42), PolyaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_draws() {
        assert_eq!(simulate(1, 1, 0, 42), PolyaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1, 1, 50, 42);
        let b = simulate(1, 1, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn total_balls_correct() {
        let v = simulate(1, 1, 100, 42);
        if let PolyaVerdict::Ok { total_balls, .. } = v {
            assert_eq!(total_balls, 102);
        }
    }

    #[test]
    fn red_fraction_in_zero_one() {
        let v = simulate(1, 1, 100, 42);
        if let PolyaVerdict::Ok {
            red_fraction_x1000, ..
        } = v
        {
            assert!(red_fraction_x1000 <= 1000);
        }
    }

    #[test]
    fn pure_red_stays_pure() {
        let v = simulate(1, 0, 100, 42);
        if let PolyaVerdict::Ok {
            red_fraction_x1000, ..
        } = v
        {
            assert_eq!(red_fraction_x1000, 1000);
        }
    }

    #[test]
    fn pure_black_stays_pure() {
        let v = simulate(0, 1, 100, 42);
        if let PolyaVerdict::Ok {
            red_fraction_x1000, ..
        } = v
        {
            assert_eq!(red_fraction_x1000, 0);
        }
    }

    #[test]
    fn biased_start_biases_outcome() {
        // Start 10:1 → red should dominate.
        let v = simulate(10, 1, 100, 42);
        if let PolyaVerdict::Ok {
            red_fraction_x1000, ..
        } = v
        {
            assert!(red_fraction_x1000 > 700);
        }
    }

    #[test]
    fn many_draws_handled() {
        let v = simulate(1, 1, 10_000, 42);
        if let PolyaVerdict::Ok { total_balls, .. } = v {
            assert_eq!(total_balls, 10_002);
        }
    }

    #[test]
    fn different_seeds_different_outcomes() {
        // Symmetric urn: outcomes vary widely between seeds.
        let a = simulate(1, 1, 200, 42);
        let b = simulate(1, 1, 200, 999);
        assert!(a != b);
    }

    #[test]
    fn small_initial_handled() {
        let v = simulate(1, 1, 1, 42);
        if let PolyaVerdict::Ok { total_balls, .. } = v {
            assert_eq!(total_balls, 3);
        }
    }
}
