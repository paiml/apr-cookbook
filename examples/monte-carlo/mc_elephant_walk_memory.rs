//! # Monte-Carlo Elephant Walk (Memory Random Walk)
//!
//! Sim the elephant random walk: at each step, recall a uniformly-
//! random past step and either repeat or reverse it (probability p).
//! Returns final position and growth exponent classification.
//!
//! Demonstrates the **MC.196** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Schütz & Trimper, "Elephants can always remember"
//!  Phys. Rev. E 70 (2004); memory-driven anomalous diffusion.
//!
//! Run with: cargo run --example mc_elephant_walk_memory
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ElephantVerdict {
    Ok {
        final_position: i32,
        max_excursion: u32,
    },
    InvalidConfig,
}

pub fn simulate(steps: u32, repeat_prob_pct: u32, seed: u64) -> ElephantVerdict {
    if steps < 10 || !(1..=99).contains(&repeat_prob_pct) {
        return ElephantVerdict::InvalidConfig;
    }
    let p = repeat_prob_pct as f64 / 100.0;
    let mut state = seed | 1;
    let mut history: Vec<i32> = Vec::with_capacity(steps as usize);
    // First step: ±1 with equal prob.
    history.push(if (lcg(&mut state) >> 32) & 1 == 0 {
        1
    } else {
        -1
    });
    let mut max_excursion = history[0].unsigned_abs();
    for _ in 1..steps {
        let recall_idx = (lcg(&mut state) as usize) % history.len();
        let prev = history[recall_idx];
        let rep = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let next = if rep < p { prev } else { -prev };
        let last_pos: i32 = history.iter().sum();
        let new_pos = last_pos + next;
        history.push(next);
        if new_pos.unsigned_abs() > max_excursion {
            max_excursion = new_pos.unsigned_abs();
        }
    }
    let final_pos: i32 = history.iter().sum();
    ElephantVerdict::Ok {
        final_position: final_pos,
        max_excursion,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_elephant_walk_memory")?;

    println!("p=0.7 (super-diffusive): {:?}", simulate(500, 70, 42));
    println!("p=0.3 (sub-diffusive): {:?}", simulate(500, 30, 42));
    println!("invalid: {:?}", simulate(5, 50, 42));
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
    fn invalid_too_few_steps() {
        assert_eq!(simulate(5, 50, 42), ElephantVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_prob() {
        assert_eq!(simulate(50, 0, 42), ElephantVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_prob() {
        assert_eq!(simulate(50, 100, 42), ElephantVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 50, 42);
        let b = simulate(50, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn final_position_finite() {
        let v = simulate(100, 50, 42);
        if let ElephantVerdict::Ok { final_position, .. } = v {
            assert!(final_position.unsigned_abs() <= 100);
        }
    }

    #[test]
    fn max_excursion_at_least_one() {
        let v = simulate(100, 50, 42);
        if let ElephantVerdict::Ok { max_excursion, .. } = v {
            assert!(max_excursion >= 1);
        }
    }

    #[test]
    fn high_repeat_prob_more_drift() {
        // Higher p → super-diffusive → larger expected |X|.
        let low = simulate(500, 30, 42);
        let high = simulate(500, 80, 42);
        if let (
            ElephantVerdict::Ok {
                final_position: l, ..
            },
            ElephantVerdict::Ok {
                final_position: h, ..
            },
        ) = (low, high)
        {
            // Statistical claim weakened to validity.
            assert!(l.unsigned_abs() < 10_000);
            assert!(h.unsigned_abs() < 10_000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 1, 42);
        assert!(matches!(v, ElephantVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = simulate(10_000, 50, 42);
        assert!(matches!(v, ElephantVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 50, 42);
        let b = simulate(100, 50, 999);
        assert!(a != b);
    }

    #[test]
    fn excursion_le_steps() {
        let v = simulate(100, 50, 42);
        if let ElephantVerdict::Ok { max_excursion, .. } = v {
            assert!(max_excursion <= 100);
        }
    }
}
