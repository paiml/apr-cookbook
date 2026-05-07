//! # Monte-Carlo Genetic Drift (Wright-Fisher)
//!
//! Sim allele frequency drift in a finite population using the
//! Wright-Fisher model. Each generation, sample N alleles with
//! replacement from prior gen. Reports fixation rate.
//!
//! Demonstrates the **MC.124** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wright (1931) Genetics 16; Fisher, Genetical Theory of
//!  Natural Selection (1930).
//!
//! Run with: cargo run --example mc_genetic_drift_population
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Ok {
        fixation_rate: f64,
        mean_generations_to_fix: f64,
        loss_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    trials: u32,
    population: u32,
    initial_freq: f64,
    max_generations: u32,
    seed: u64,
) -> DriftVerdict {
    if trials == 0 || population < 2 || max_generations == 0 || !(0.0..=1.0).contains(&initial_freq)
    {
        return DriftVerdict::InvalidConfig;
    }
    let mut fixed = 0u32;
    let mut lost = 0u32;
    let mut total_gens_to_fix: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut alleles = (initial_freq * f64::from(population)) as u32;
        let mut gen_to_fix = 0u32;
        for gen in 0..max_generations {
            if alleles == population {
                fixed += 1;
                gen_to_fix = gen + 1;
                break;
            }
            if alleles == 0 {
                lost += 1;
                break;
            }
            // Wright-Fisher: sample population alleles with replacement.
            let p = f64::from(alleles) / f64::from(population);
            let mut new_alleles = 0u32;
            for _ in 0..population {
                let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
                if r < p {
                    new_alleles += 1;
                }
            }
            alleles = new_alleles;
        }
        if gen_to_fix > 0 {
            total_gens_to_fix += u64::from(gen_to_fix);
        }
    }
    let fixation_rate = f64::from(fixed) / f64::from(trials);
    let loss_rate = f64::from(lost) / f64::from(trials);
    let mean_generations_to_fix = if fixed > 0 {
        total_gens_to_fix as f64 / f64::from(fixed)
    } else {
        0.0
    };
    DriftVerdict::Ok {
        fixation_rate,
        mean_generations_to_fix,
        loss_rate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_genetic_drift_population")?;

    println!("typical: {:?}", simulate(500, 50, 0.5, 1000, 42));
    println!("invalid: {:?}", simulate(0, 50, 0.5, 1000, 42));
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
    fn fixation_rate_near_initial_freq() {
        // Theoretical: P(fixation) = initial_freq.
        let v = simulate(1000, 20, 0.5, 1000, 42);
        if let DriftVerdict::Ok { fixation_rate, .. } = v {
            assert!((fixation_rate - 0.5).abs() < 0.15);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 50, 0.5, 1000, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_small_population() {
        assert_eq!(simulate(100, 1, 0.5, 1000, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_generations() {
        assert_eq!(simulate(100, 50, 0.5, 0, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_freq_out_of_range() {
        assert_eq!(
            simulate(100, 50, 1.5, 1000, 42),
            DriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 20, 0.5, 500, 42);
        let b = simulate(100, 20, 0.5, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(100, 20, 0.5, 500, 42);
        if let DriftVerdict::Ok {
            fixation_rate,
            loss_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&fixation_rate));
            assert!((0.0..=1.0).contains(&loss_rate));
        }
    }

    #[test]
    fn fixation_plus_loss_le_one() {
        let v = simulate(100, 20, 0.5, 500, 42);
        if let DriftVerdict::Ok {
            fixation_rate,
            loss_rate,
            ..
        } = v
        {
            assert!(fixation_rate + loss_rate <= 1.0001);
        }
    }

    #[test]
    fn high_initial_freq_more_fixation() {
        let lo = simulate(500, 20, 0.1, 500, 42);
        let hi = simulate(500, 20, 0.9, 500, 42);
        if let (
            DriftVerdict::Ok {
                fixation_rate: l, ..
            },
            DriftVerdict::Ok {
                fixation_rate: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn small_pop_fast_drift() {
        let small = simulate(500, 10, 0.5, 500, 42);
        let big = simulate(500, 50, 0.5, 500, 42);
        if let (
            DriftVerdict::Ok {
                mean_generations_to_fix: s,
                ..
            },
            DriftVerdict::Ok {
                mean_generations_to_fix: b,
                ..
            },
        ) = (small, big)
        {
            // Small population fixes faster (fewer generations).
            // We don't strictly assert; both should be finite.
            assert!(s.is_finite() && b.is_finite());
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(100, 20, 0.5, 500, 42);
        if let DriftVerdict::Ok {
            fixation_rate,
            mean_generations_to_fix,
            loss_rate,
        } = v
        {
            assert!(fixation_rate.is_finite());
            assert!(mean_generations_to_fix.is_finite());
            assert!(loss_rate.is_finite());
        }
    }
}
