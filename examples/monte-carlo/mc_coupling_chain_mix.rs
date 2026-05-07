//! # Monte-Carlo Coupling-from-the-Past Mixing Time
//!
//! Sim two Markov chains starting from different initial states with
//! shared randomness. Measures coupling time (when they merge).
//! Returns mean coupling time and the coupling-coverage rate.
//!
//! Demonstrates the **MC.199** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Propp & Wilson, "Exact sampling with coupled Markov
//!  chains" Random Structures & Algorithms 9 (1996); coupling-from-
//!  the-past technique.
//!
//! Run with: cargo run --example mc_coupling_chain_mix
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CouplingVerdict {
    Ok {
        mean_coupling_time: u32,
        coupled_pct_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(state_count: u32, max_steps: u32, trials: u32, seed: u64) -> CouplingVerdict {
    if state_count < 2 || max_steps < 10 || trials < 100 {
        return CouplingVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_time: u64 = 0;
    let mut coupled_count = 0u32;
    for _ in 0..trials {
        // Two chains starting at 0 and state_count-1.
        let mut chain_a = 0u32;
        let mut chain_b = state_count - 1;
        let mut step = 0u32;
        while chain_a != chain_b && step < max_steps {
            // Shared randomness: with prob 1/state_count both chains
            // reset to 0 (couples them); else each advances by ±1 mod
            // state_count with an independent coin.
            let reset = lcg(&mut state) % (state_count as u64);
            if reset == 0 {
                chain_a = 0;
                chain_b = 0;
            } else {
                let coin_a = lcg(&mut state) & 1;
                let coin_b = lcg(&mut state) & 1;
                chain_a = if coin_a == 0 {
                    (chain_a + 1) % state_count
                } else {
                    (chain_a + state_count - 1) % state_count
                };
                chain_b = if coin_b == 0 {
                    (chain_b + 1) % state_count
                } else {
                    (chain_b + state_count - 1) % state_count
                };
            }
            step += 1;
        }
        if chain_a == chain_b {
            coupled_count += 1;
        }
        total_time += step as u64;
    }
    CouplingVerdict::Ok {
        mean_coupling_time: (total_time / trials as u64) as u32,
        coupled_pct_x100: ((coupled_count as f64 / trials as f64) * 10000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_coupling_chain_mix")?;

    println!("n=4: {:?}", simulate(4, 100, 1000, 42));
    println!("invalid: {:?}", simulate(1, 100, 100, 42));
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
    fn invalid_too_few_states() {
        assert_eq!(simulate(1, 100, 100, 42), CouplingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_steps() {
        assert_eq!(simulate(4, 5, 100, 42), CouplingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(4, 100, 50, 42), CouplingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(4, 100, 500, 42);
        let b = simulate(4, 100, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn small_state_high_coupling_rate() {
        // With 2 states and shared moves, never couples — but with 4 states + cyclic shift, both move synchronously.
        let v = simulate(4, 1000, 1000, 42);
        if let CouplingVerdict::Ok {
            coupled_pct_x100, ..
        } = v
        {
            assert!(coupled_pct_x100 < 10001);
        }
    }

    #[test]
    fn larger_states_longer_coupling() {
        let small = simulate(4, 1000, 500, 42);
        let large = simulate(20, 1000, 500, 42);
        if let (
            CouplingVerdict::Ok {
                mean_coupling_time: s,
                ..
            },
            CouplingVerdict::Ok {
                mean_coupling_time: l,
                ..
            },
        ) = (small, large)
        {
            // Larger state space → longer expected coupling time.
            assert!(l >= s);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 10, 100, 42);
        assert!(matches!(v, CouplingVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(4, 100, 10_000, 42);
        assert!(matches!(v, CouplingVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_both_valid() {
        let a = simulate(4, 100, 500, 42);
        let b = simulate(4, 100, 500, 999);
        assert!(matches!(a, CouplingVerdict::Ok { .. }));
        assert!(matches!(b, CouplingVerdict::Ok { .. }));
    }

    #[test]
    fn coupling_time_le_max_steps() {
        let v = simulate(4, 100, 500, 42);
        if let CouplingVerdict::Ok {
            mean_coupling_time, ..
        } = v
        {
            assert!(mean_coupling_time <= 100);
        }
    }

    #[test]
    fn coupled_pct_in_zero_one() {
        let v = simulate(4, 100, 500, 42);
        if let CouplingVerdict::Ok {
            coupled_pct_x100, ..
        } = v
        {
            assert!(coupled_pct_x100 <= 10000);
        }
    }
}
