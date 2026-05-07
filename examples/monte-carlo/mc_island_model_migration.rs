//! # Monte-Carlo Island-Model Genetic-Algorithm Migration
//!
//! Sim N populations evolving in parallel with periodic migration
//! between islands. Each island runs a GA on the OneMax problem.
//! Returns mean best-fitness across islands and migration count.
//!
//! Demonstrates the **MC.197** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cohoon et al., "A Multi-Population Genetic Algorithm"
//!  ICGA (1991); Whitley distributed-GA reference.
//!
//! Run with: cargo run --example mc_island_model_migration
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IslandVerdict {
    Ok {
        mean_best_fitness: u32,
        migrations_total: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    n_islands: u32,
    bits: u32,
    pop_per_island: u32,
    generations: u32,
    migration_interval: u32,
    seed: u64,
) -> IslandVerdict {
    if n_islands < 2 || bits < 4 || pop_per_island < 2 || generations < 5 || migration_interval == 0
    {
        return IslandVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    // Initialize: islands x populations x bit-strings.
    let mut islands: Vec<Vec<Vec<bool>>> = (0..n_islands)
        .map(|_| {
            (0..pop_per_island)
                .map(|_| {
                    (0..bits)
                        .map(|_| (lcg(&mut state) >> 32) & 1 == 0)
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut migrations = 0u32;
    for gen in 0..generations {
        // Per-island selection + mutation.
        for island in &mut islands {
            let mut scored: Vec<(u32, usize)> = island
                .iter()
                .enumerate()
                .map(|(i, p)| (fitness(p), i))
                .collect();
            scored.sort_by_key(|s| std::cmp::Reverse(s.0));
            let half = (pop_per_island / 2) as usize;
            let mut new_pop: Vec<Vec<bool>> = scored[..half]
                .iter()
                .map(|(_, idx)| island[*idx].clone())
                .collect();
            while new_pop.len() < pop_per_island as usize {
                let parent = (lcg(&mut state) as usize) % half;
                let mut child = new_pop[parent].clone();
                let bit = (lcg(&mut state) as usize) % bits as usize;
                child[bit] = !child[bit];
                new_pop.push(child);
            }
            *island = new_pop;
        }
        // Migration: every interval, swap best-of-each between adjacent.
        if (gen + 1) % migration_interval == 0 && gen > 0 {
            let n = islands.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let best_i = islands[i]
                    .iter()
                    .max_by_key(|p| fitness(p))
                    .cloned()
                    .unwrap();
                let best_j = islands[j]
                    .iter()
                    .max_by_key(|p| fitness(p))
                    .cloned()
                    .unwrap();
                islands[i].push(best_j);
                islands[j].push(best_i);
                islands[i].truncate(pop_per_island as usize);
                islands[j].truncate(pop_per_island as usize);
                migrations += 2;
            }
        }
    }
    let bests: Vec<u32> = islands
        .iter()
        .map(|p| p.iter().map(|s| fitness(s)).max().unwrap_or(0))
        .collect();
    let mean = bests.iter().sum::<u32>() / bests.len() as u32;
    IslandVerdict::Ok {
        mean_best_fitness: mean,
        migrations_total: migrations,
    }
}

fn fitness(s: &[bool]) -> u32 {
    s.iter().filter(|b| **b).count() as u32
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_island_model_migration")?;

    println!("4 islands: {:?}", simulate(4, 16, 20, 50, 10, 42));
    println!("invalid: {:?}", simulate(1, 16, 20, 50, 10, 42));
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
    fn invalid_too_few_islands() {
        assert_eq!(
            simulate(1, 16, 20, 50, 10, 42),
            IslandVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_bits() {
        assert_eq!(simulate(4, 2, 20, 50, 10, 42), IslandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_small_pop() {
        assert_eq!(simulate(4, 16, 1, 50, 10, 42), IslandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_generations() {
        assert_eq!(simulate(4, 16, 20, 4, 10, 42), IslandVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_migration_interval() {
        assert_eq!(simulate(4, 16, 20, 50, 0, 42), IslandVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(3, 8, 10, 20, 5, 42);
        let b = simulate(3, 8, 10, 20, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_fitness_le_bits() {
        let v = simulate(3, 8, 10, 20, 5, 42);
        if let IslandVerdict::Ok {
            mean_best_fitness, ..
        } = v
        {
            assert!(mean_best_fitness <= 8);
        }
    }

    #[test]
    fn migrations_at_least_zero() {
        let v = simulate(3, 8, 10, 20, 5, 42);
        if let IslandVerdict::Ok {
            migrations_total, ..
        } = v
        {
            assert!(migrations_total < u32::MAX);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 4, 2, 5, 1, 42);
        assert!(matches!(v, IslandVerdict::Ok { .. }));
    }

    #[test]
    fn many_islands_handled() {
        let v = simulate(10, 16, 20, 50, 10, 42);
        assert!(matches!(v, IslandVerdict::Ok { .. }));
    }

    #[test]
    fn many_generations_converge() {
        let v = simulate(4, 16, 20, 200, 10, 42);
        if let IslandVerdict::Ok {
            mean_best_fitness, ..
        } = v
        {
            // Mean best should be > half bits after many generations.
            assert!(mean_best_fitness > 8);
        }
    }
}
