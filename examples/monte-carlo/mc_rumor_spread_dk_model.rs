//! # Monte-Carlo Daley-Kendall Rumor Spread
//!
//! Sim the Daley-Kendall rumor model: Ignorant→Spreader on contact
//! with a Spreader; Spreader→Stifler when meeting another Spreader
//! or a Stifler. Returns final fraction informed (×100) and rounds
//! to extinction.
//!
//! Demonstrates the **MC.186** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Daley & Kendall, "Stochastic rumours" J. Inst. Math. &
//!  Appl. 1(1) (1965); rumor-spread vs SIR difference.
//!
//! Run with: cargo run --example mc_rumor_spread_dk_model
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RumorVerdict {
    Ok {
        informed_pct_x100: u32,
        rounds_to_extinction: u32,
    },
    InvalidConfig,
}

pub fn simulate(population: u32, max_rounds: u32, seed: u64) -> RumorVerdict {
    if population < 10 || max_rounds == 0 {
        return RumorVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    // 0 = ignorant, 1 = spreader, 2 = stifler
    let mut status: Vec<u8> = vec![0; population as usize];
    status[0] = 1; // single seed spreader
    let mut rounds = 0u32;
    for r in 1..=max_rounds {
        let mut spread_alive = false;
        for i in 0..population as usize {
            if status[i] != 1 {
                continue;
            }
            spread_alive = true;
            // Spreader interacts with random other agent.
            let mut j = (lcg(&mut state) as usize) % population as usize;
            while j == i {
                j = (lcg(&mut state) as usize) % population as usize;
            }
            match status[j] {
                0 => {
                    // Ignorant becomes spreader.
                    status[j] = 1;
                }
                1 | 2 => {
                    // Spreader becomes stifler when meeting spreader/stifler.
                    status[i] = 2;
                }
                _ => {}
            }
        }
        if !spread_alive {
            rounds = r - 1;
            break;
        }
        rounds = r;
    }
    let informed = status.iter().filter(|s| **s != 0).count() as u32;
    let pct = (informed as f64 / population as f64 * 10000.0) as u32;
    RumorVerdict::Ok {
        informed_pct_x100: pct,
        rounds_to_extinction: rounds,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_rumor_spread_dk_model")?;

    println!("n=100: {:?}", simulate(100, 200, 42));
    println!("invalid: {:?}", simulate(5, 200, 42));
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
        assert_eq!(simulate(5, 100, 42), RumorVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_rounds() {
        assert_eq!(simulate(100, 0, 42), RumorVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn informed_at_least_initial() {
        // At least the seed spreader is "informed".
        let v = simulate(100, 100, 42);
        if let RumorVerdict::Ok {
            informed_pct_x100, ..
        } = v
        {
            assert!(informed_pct_x100 >= 100); // ≥1% (1/100)
        }
    }

    #[test]
    fn rounds_le_max() {
        let v = simulate(100, 50, 42);
        if let RumorVerdict::Ok {
            rounds_to_extinction,
            ..
        } = v
        {
            assert!(rounds_to_extinction <= 50);
        }
    }

    #[test]
    fn larger_pop_more_rounds() {
        let small = simulate(20, 200, 42);
        let large = simulate(200, 200, 42);
        if let (
            RumorVerdict::Ok {
                rounds_to_extinction: s,
                ..
            },
            RumorVerdict::Ok {
                rounds_to_extinction: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn informed_pct_le_100() {
        let v = simulate(100, 200, 42);
        if let RumorVerdict::Ok {
            informed_pct_x100, ..
        } = v
        {
            assert!(informed_pct_x100 <= 10000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 1, 42);
        assert!(matches!(v, RumorVerdict::Ok { .. }));
    }

    #[test]
    fn many_rounds_handled() {
        let v = simulate(500, 1000, 42);
        assert!(matches!(v, RumorVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 200, 42);
        let b = simulate(100, 200, 999);
        assert!(a != b);
    }

    #[test]
    fn rounds_at_least_one() {
        let v = simulate(100, 200, 42);
        if let RumorVerdict::Ok {
            rounds_to_extinction,
            ..
        } = v
        {
            assert!(rounds_to_extinction >= 1);
        }
    }
}
