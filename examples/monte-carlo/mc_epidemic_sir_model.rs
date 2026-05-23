//! # Monte-Carlo Epidemic SIR Model
//!
//! Sim the classical SIR epidemic compartmental model: Susceptible →
//! Infected → Recovered with deterministic ODE-like discrete update.
//! Returns peak infected count and total recovered at end.
//!
//! Demonstrates the **MC.175** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kermack & McKendrick, "A Contribution to the Mathematical
//!  Theory of Epidemics" Proc. Roy. Soc. A 115 (1927).
//!
//! Run with: cargo run --example mc_epidemic_sir_model
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SirVerdict {
    Ok {
        peak_infected: u32,
        total_recovered: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    population: u32,
    initial_infected: u32,
    beta_x100: u32,
    gamma_x100: u32,
    days: u32,
) -> SirVerdict {
    if population < 100
        || initial_infected == 0
        || initial_infected >= population
        || beta_x100 == 0
        || gamma_x100 == 0
        || days == 0
    {
        return SirVerdict::InvalidConfig;
    }
    let beta = beta_x100 as f64 / 100.0;
    let gamma = gamma_x100 as f64 / 100.0;
    let n = population as f64;
    let mut s = (population - initial_infected) as f64;
    let mut i = initial_infected as f64;
    let mut r = 0.0f64;
    let mut peak = initial_infected;
    for _ in 0..days {
        let new_infections = beta * s * i / n;
        let new_recoveries = gamma * i;
        s -= new_infections;
        i += new_infections - new_recoveries;
        r += new_recoveries;
        if i < 0.0 {
            i = 0.0;
        }
        if i as u32 > peak {
            peak = i as u32;
        }
    }
    SirVerdict::Ok {
        peak_infected: peak,
        total_recovered: r as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_epidemic_sir_model")?;

    println!("R0=2: {:?}", simulate(10_000, 10, 30, 15, 100));
    println!("invalid: {:?}", simulate(50, 10, 30, 15, 100));
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
    fn invalid_too_small_pop() {
        assert_eq!(simulate(50, 10, 30, 15, 100), SirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_initial() {
        assert_eq!(simulate(1000, 0, 30, 15, 100), SirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_initial_ge_pop() {
        assert_eq!(simulate(1000, 1000, 30, 15, 100), SirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_beta() {
        assert_eq!(simulate(1000, 10, 0, 15, 100), SirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_gamma() {
        assert_eq!(simulate(1000, 10, 30, 0, 100), SirVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_days() {
        assert_eq!(simulate(1000, 10, 30, 15, 0), SirVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 10, 30, 15, 50);
        let b = simulate(1000, 10, 30, 15, 50);
        assert_eq!(a, b);
    }

    #[test]
    fn high_R0_high_peak() {
        // R0 = β/γ. High R0 → bigger epidemic peak.
        let low = simulate(10_000, 10, 15, 15, 100);
        let high = simulate(10_000, 10, 60, 15, 100);
        if let (
            SirVerdict::Ok {
                peak_infected: l, ..
            },
            SirVerdict::Ok {
                peak_infected: h, ..
            },
        ) = (low, high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn peak_at_least_initial() {
        let v = simulate(10_000, 100, 30, 15, 100);
        if let SirVerdict::Ok { peak_infected, .. } = v {
            assert!(peak_infected >= 100);
        }
    }

    #[test]
    fn total_recovered_le_population() {
        let v = simulate(10_000, 10, 30, 15, 100);
        if let SirVerdict::Ok {
            total_recovered, ..
        } = v
        {
            assert!(total_recovered <= 10_000);
        }
    }

    #[test]
    fn long_simulation_handled() {
        let v = simulate(10_000, 10, 30, 15, 1000);
        assert!(matches!(v, SirVerdict::Ok { .. }));
    }

    #[test]
    fn peak_le_population() {
        let v = simulate(10_000, 10, 30, 15, 100);
        if let SirVerdict::Ok { peak_infected, .. } = v {
            assert!(peak_infected <= 10_000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(100, 1, 1, 1, 1);
        assert!(matches!(v, SirVerdict::Ok { .. }));
    }
}
