//! # Monte-Carlo 0/1 Knapsack via Random Sampling
//!
//! Sample N random subsets of items, keeping the highest-value subset
//! that fits within the knapsack capacity. Returns best value and a
//! count of feasible samples.
//!
//! Demonstrates the **MC.134** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kellerer, Pferschy & Pisinger, Knapsack Problems (2004);
//!  Karp 21 NP-complete problems (1972).
//!
//! Run with: cargo run --example mc_knapsack_random
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KnapsackVerdict {
    Ok {
        best_value: u32,
        feasible_samples: u32,
    },
    InvalidConfig,
}

pub fn simulate(items: &[(u32, u32)], capacity: u32, samples: u32, seed: u64) -> KnapsackVerdict {
    if items.is_empty() || capacity == 0 || samples == 0 {
        return KnapsackVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut best_value = 0u32;
    let mut feasible = 0u32;
    for _ in 0..samples {
        let mut total_w = 0u32;
        let mut total_v = 0u32;
        for &(w, v) in items {
            let bit = (lcg(&mut state) >> 32) % 2 == 0;
            if bit {
                total_w = total_w.saturating_add(w);
                total_v = total_v.saturating_add(v);
            }
        }
        if total_w <= capacity {
            feasible += 1;
            if total_v > best_value {
                best_value = total_v;
            }
        }
    }
    KnapsackVerdict::Ok {
        best_value,
        feasible_samples: feasible,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_knapsack_random")?;

    let items = [(2, 3), (3, 4), (4, 5), (5, 6)];
    println!("knapsack-cap-7: {:?}", simulate(&items, 7, 1000, 42));
    println!("invalid: {:?}", simulate(&[], 7, 1000, 42));
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
    fn finds_optimal_for_small_problem() {
        // Known optimum for items=[(2,3),(3,4),(4,5)], cap=5 is 7 (items 1+2)
        let items = [(2, 3), (3, 4), (4, 5)];
        let v = simulate(&items, 5, 5000, 42);
        if let KnapsackVerdict::Ok { best_value, .. } = v {
            assert_eq!(best_value, 7);
        }
    }

    #[test]
    fn invalid_empty_items() {
        assert_eq!(simulate(&[], 10, 100, 42), KnapsackVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(
            simulate(&[(1, 1)], 0, 100, 42),
            KnapsackVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(&[(1, 1)], 10, 0, 42),
            KnapsackVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let items = [(1, 1), (2, 2)];
        let a = simulate(&items, 3, 100, 42);
        let b = simulate(&items, 3, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn best_value_le_total_value() {
        let items = [(1, 5), (2, 10)];
        let v = simulate(&items, 10, 100, 42);
        if let KnapsackVerdict::Ok { best_value, .. } = v {
            assert!(best_value <= 15);
        }
    }

    #[test]
    fn feasible_count_le_samples() {
        let items = [(1, 1)];
        let v = simulate(&items, 10, 100, 42);
        if let KnapsackVerdict::Ok {
            feasible_samples, ..
        } = v
        {
            assert!(feasible_samples <= 100);
        }
    }

    #[test]
    fn larger_capacity_more_value() {
        let items = [(2, 3), (3, 4), (4, 5)];
        let small = simulate(&items, 3, 1000, 42);
        let large = simulate(&items, 9, 1000, 42);
        if let (
            KnapsackVerdict::Ok { best_value: s, .. },
            KnapsackVerdict::Ok { best_value: l, .. },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn single_item_handled() {
        let v = simulate(&[(1, 5)], 10, 100, 42);
        if let KnapsackVerdict::Ok { best_value, .. } = v {
            assert_eq!(best_value, 5);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(u32, u32)> = (1..=10).map(|i| (i, i + 1)).collect();
        let v = simulate(&items, 30, 1000, 42);
        assert!(matches!(v, KnapsackVerdict::Ok { .. }));
    }

    #[test]
    fn impossible_capacity_zero_value() {
        // All items too heavy
        let items = [(100, 5), (200, 10)];
        let v = simulate(&items, 1, 100, 42);
        if let KnapsackVerdict::Ok { best_value, .. } = v {
            assert_eq!(best_value, 0);
        }
    }
}
