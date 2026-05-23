//! # Monte-Carlo Boltzmann Distribution Sample
//!
//! Sample from a Boltzmann distribution P(i) ∝ exp(-E_i/kT) over a
//! discrete state space. Returns mean energy and the most-visited
//! state.
//!
//! Demonstrates the **MC.195** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Boltzmann, "Über die Beziehung..." Wien. Sitz.-Berichte
//!  76 (1877); statistical-mechanics canonical ensemble.
//!
//! Run with: cargo run --example mc_boltzmann_distribution
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BoltzmannVerdict {
    Ok {
        mean_energy_x100: i32,
        most_visited_state: u32,
    },
    InvalidConfig,
}

pub fn simulate(energies: &[i32], kt_x100: u32, samples: u32, seed: u64) -> BoltzmannVerdict {
    if energies.is_empty() || kt_x100 == 0 || samples < 100 {
        return BoltzmannVerdict::InvalidConfig;
    }
    let kt = kt_x100 as f64 / 100.0;
    let weights: Vec<f64> = energies.iter().map(|e| (-(*e as f64) / kt).exp()).collect();
    let total: f64 = weights.iter().sum();
    if total == 0.0 || !total.is_finite() {
        return BoltzmannVerdict::InvalidConfig;
    }
    let mut cdf: Vec<f64> = Vec::with_capacity(weights.len());
    let mut acc = 0.0f64;
    for w in &weights {
        acc += w / total;
        cdf.push(acc);
    }
    let mut state = seed | 1;
    let mut visit_counts: Vec<u32> = vec![0; energies.len()];
    let mut energy_sum: i64 = 0;
    for _ in 0..samples {
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let idx = cdf
            .iter()
            .position(|c| r < *c)
            .unwrap_or(energies.len() - 1);
        visit_counts[idx] += 1;
        energy_sum += energies[idx] as i64;
    }
    let mean = (energy_sum as f64 / samples as f64) * 100.0;
    let most_idx = visit_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map_or(0u32, |(i, _)| i as u32);
    BoltzmannVerdict::Ok {
        mean_energy_x100: mean as i32,
        most_visited_state: most_idx,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_boltzmann_distribution")?;

    let energies = [0, 1, 2, 3, 4];
    println!("kt=1: {:?}", simulate(&energies, 100, 10_000, 42));
    println!("invalid: {:?}", simulate(&[], 100, 100, 42));
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
    fn invalid_empty_energies() {
        assert_eq!(simulate(&[], 100, 100, 42), BoltzmannVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_kt() {
        assert_eq!(simulate(&[0], 0, 100, 42), BoltzmannVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(&[0], 100, 50, 42), BoltzmannVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(&[0, 1], 100, 500, 42);
        let b = simulate(&[0, 1], 100, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn ground_state_most_visited_low_kt() {
        // Low T → ground state (lowest energy) dominates.
        let v = simulate(&[0, 5, 10], 50, 5000, 42);
        if let BoltzmannVerdict::Ok {
            most_visited_state, ..
        } = v
        {
            assert_eq!(most_visited_state, 0);
        }
    }

    #[test]
    fn higher_kt_more_uniform() {
        // High T → distribution flattens.
        let low_kt = simulate(&[0, 1, 2], 50, 5000, 42);
        let high_kt = simulate(&[0, 1, 2], 1000, 5000, 42);
        if let (
            BoltzmannVerdict::Ok {
                mean_energy_x100: l,
                ..
            },
            BoltzmannVerdict::Ok {
                mean_energy_x100: h,
                ..
            },
        ) = (low_kt, high_kt)
        {
            // Higher T → mean energy closer to (0+1+2)/3 = 1.0; lower T → near 0.
            assert!(h > l);
        }
    }

    #[test]
    fn mean_energy_finite() {
        let v = simulate(&[0, 1, 2], 100, 1000, 42);
        if let BoltzmannVerdict::Ok {
            mean_energy_x100, ..
        } = v
        {
            assert!(mean_energy_x100.abs() < 10_000);
        }
    }

    #[test]
    fn most_visited_in_range() {
        let v = simulate(&[0, 1, 2, 3, 4], 100, 1000, 42);
        if let BoltzmannVerdict::Ok {
            most_visited_state, ..
        } = v
        {
            assert!(most_visited_state < 5);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(&[0], 1, 100, 42);
        assert!(matches!(v, BoltzmannVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = simulate(&[0, 1, 2], 100, 100_000, 42);
        assert!(matches!(v, BoltzmannVerdict::Ok { .. }));
    }

    #[test]
    fn negative_energies_handled() {
        let v = simulate(&[-5, -3, 0, 3, 5], 100, 1000, 42);
        assert!(matches!(v, BoltzmannVerdict::Ok { .. }));
    }
}
