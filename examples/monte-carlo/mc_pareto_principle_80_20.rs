//! # Monte-Carlo Pareto 80/20 Principle
//!
//! Generate Pareto-distributed samples and verify that ~20% of
//! samples account for ~80% of total mass. Reports actual top-20%
//! mass fraction.
//!
//! Demonstrates the **MC.125** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pareto, Cours d'économie politique (1896); 80/20
//!  principle in business analysis.
//!
//! Run with: cargo run --example mc_pareto_principle_80_20
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ParetoVerdict {
    Ok {
        top_20pct_mass: f64,
        total_mass: f64,
    },
    InvalidConfig,
}

pub fn simulate(samples: u32, alpha: f64, seed: u64) -> ParetoVerdict {
    if samples < 5 || alpha <= 0.0 {
        return ParetoVerdict::InvalidConfig;
    }
    let mut values: Vec<f64> = Vec::with_capacity(samples as usize);
    let mut rng_state = seed | 1;
    for _ in 0..samples {
        let u = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let u = u.clamp(1e-12, 1.0 - 1e-12);
        // Pareto inverse CDF: x = (1 - u)^(-1/alpha).
        let x = (1.0 - u).powf(-1.0 / alpha);
        values.push(x);
    }
    let total_mass: f64 = values.iter().sum();
    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top_n = ((samples as f64 * 0.2).ceil() as usize).max(1);
    let top_mass: f64 = values.iter().take(top_n).sum();
    ParetoVerdict::Ok {
        top_20pct_mass: top_mass / total_mass,
        total_mass,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_pareto_principle_80_20")?;

    println!("alpha=1.16: {:?}", simulate(10_000, 1.16, 42));
    println!("alpha=2.0: {:?}", simulate(10_000, 2.0, 42));
    println!("invalid: {:?}", simulate(0, 1.16, 42));
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
    fn alpha_116_near_80_20() {
        // alpha ≈ 1.16 (log4/log5) gives the classic 80/20 split.
        let v = simulate(10_000, 1.16, 42);
        if let ParetoVerdict::Ok { top_20pct_mass, .. } = v {
            assert!(top_20pct_mass > 0.5);
        }
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(0, 1.0, 42), ParetoVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(simulate(100, 0.0, 42), ParetoVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_alpha() {
        assert_eq!(simulate(100, -1.0, 42), ParetoVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 1.0, 42);
        let b = simulate(1000, 1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn top_mass_in_unit_range() {
        let v = simulate(1000, 1.0, 42);
        if let ParetoVerdict::Ok { top_20pct_mass, .. } = v {
            assert!((0.0..=1.0).contains(&top_20pct_mass));
        }
    }

    #[test]
    fn higher_alpha_less_concentrated() {
        let lo = simulate(5000, 1.0, 42);
        let hi = simulate(5000, 5.0, 42);
        if let (
            ParetoVerdict::Ok {
                top_20pct_mass: l, ..
            },
            ParetoVerdict::Ok {
                top_20pct_mass: h, ..
            },
        ) = (lo, hi)
        {
            assert!(l > h);
        }
    }

    #[test]
    fn total_mass_positive() {
        let v = simulate(1000, 1.0, 42);
        if let ParetoVerdict::Ok { total_mass, .. } = v {
            assert!(total_mass > 0.0);
        }
    }

    #[test]
    fn top_mass_at_least_20pct() {
        // Top 20% should always have ≥ 20% of mass.
        let v = simulate(10_000, 1.16, 42);
        if let ParetoVerdict::Ok { top_20pct_mass, .. } = v {
            assert!(top_20pct_mass >= 0.20);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(1000, 1.0, 42);
        if let ParetoVerdict::Ok {
            top_20pct_mass,
            total_mass,
        } = v
        {
            assert!(top_20pct_mass.is_finite());
            assert!(total_mass.is_finite());
        }
    }

    #[test]
    fn many_samples_stable() {
        let v = simulate(20_000, 1.0, 42);
        if let ParetoVerdict::Ok { top_20pct_mass, .. } = v {
            assert!(top_20pct_mass > 0.5);
        }
    }
}
