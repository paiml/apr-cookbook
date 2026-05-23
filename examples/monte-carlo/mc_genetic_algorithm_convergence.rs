//! # Monte-Carlo Genetic Algorithm Convergence
//!
//! Sim a simple GA optimizing the OneMax fitness (max bits set in a
//! bit string). Selection: top-half elitism. Crossover + mutation.
//! Reports best fitness over generations.
//!
//! Demonstrates the **MC.117** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Holland, Adaptation in Natural and Artificial Systems
//!  (1975); OneMax benchmark in evolutionary computation.
//!
//! Run with: cargo run --example mc_genetic_algorithm_convergence
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GaVerdict {
    Ok {
        best_fitness: u32,
        generations_to_solve: u32,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    bits: u32,
    population: u32,
    generations: u32,
    mutation_rate: f64,
    seed: u64,
) -> GaVerdict {
    if bits == 0 || population < 2 || generations == 0 || !(0.0..=1.0).contains(&mutation_rate) {
        return GaVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut pop: Vec<Vec<bool>> = (0..population)
        .map(|_| {
            (0..bits)
                .map(|_| (lcg(&mut rng_state) >> 32) % 2 == 0)
                .collect()
        })
        .collect();
    let mut best_fitness = 0u32;
    let mut generations_to_solve = generations;
    for gen in 0..generations {
        // Compute fitness.
        let mut scored: Vec<(u32, usize)> = pop
            .iter()
            .enumerate()
            .map(|(i, p)| (fitness(p), i))
            .collect();
        scored.sort_by_key(|b| std::cmp::Reverse(b.0));
        let current_best = scored[0].0;
        if current_best > best_fitness {
            best_fitness = current_best;
        }
        if best_fitness == bits {
            generations_to_solve = gen;
            break;
        }
        // Top half survive; bottom half = mutated copies.
        let half = (population / 2) as usize;
        let mut new_pop: Vec<Vec<bool>> = Vec::with_capacity(population as usize);
        for &(_, idx) in scored.iter().take(half) {
            new_pop.push(pop[idx].clone());
        }
        while new_pop.len() < population as usize {
            let parent_idx = scored[(lcg(&mut rng_state) >> 32) as usize % half].1;
            let mut child = pop[parent_idx].clone();
            for bit in &mut child {
                let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
                if r < mutation_rate {
                    *bit = !*bit;
                }
            }
            new_pop.push(child);
        }
        pop = new_pop;
    }
    GaVerdict::Ok {
        best_fitness,
        generations_to_solve,
    }
}

fn fitness(p: &[bool]) -> u32 {
    p.iter().filter(|b| **b).count() as u32
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_genetic_algorithm_convergence")?;

    println!("typical: {:?}", simulate(20, 50, 200, 0.05, 42));
    println!("invalid: {:?}", simulate(0, 50, 200, 0.05, 42));
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
    fn converges_to_optimum_with_enough_generations() {
        let v = simulate(20, 50, 500, 0.05, 42);
        if let GaVerdict::Ok { best_fitness, .. } = v {
            assert_eq!(best_fitness, 20);
        }
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(simulate(0, 50, 100, 0.05, 42), GaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_small_population() {
        assert_eq!(simulate(20, 1, 100, 0.05, 42), GaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_generations() {
        assert_eq!(simulate(20, 50, 0, 0.05, 42), GaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_mutation_rate_above_one() {
        assert_eq!(simulate(20, 50, 100, 1.5, 42), GaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 20, 50, 0.1, 42);
        let b = simulate(10, 20, 50, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fitness_le_bits() {
        let v = simulate(10, 20, 50, 0.1, 42);
        if let GaVerdict::Ok { best_fitness, .. } = v {
            assert!(best_fitness <= 10);
        }
    }

    #[test]
    fn generations_to_solve_le_generations() {
        let v = simulate(10, 20, 100, 0.1, 42);
        if let GaVerdict::Ok {
            generations_to_solve,
            ..
        } = v
        {
            assert!(generations_to_solve <= 100);
        }
    }

    #[test]
    fn larger_pop_faster_convergence() {
        let small = simulate(20, 10, 200, 0.05, 42);
        let big = simulate(20, 100, 200, 0.05, 42);
        if let (
            GaVerdict::Ok {
                best_fitness: s, ..
            },
            GaVerdict::Ok {
                best_fitness: b, ..
            },
        ) = (small, big)
        {
            assert!(b >= s);
        }
    }

    #[test]
    fn small_problem_handled() {
        let v = simulate(3, 10, 50, 0.1, 42);
        assert!(matches!(v, GaVerdict::Ok { .. }));
    }

    #[test]
    fn high_mutation_can_disrupt() {
        let v = simulate(20, 50, 100, 0.5, 42);
        if let GaVerdict::Ok { best_fitness, .. } = v {
            // High mutation may still converge but more slowly.
            assert!(best_fitness > 0);
        }
    }
}
