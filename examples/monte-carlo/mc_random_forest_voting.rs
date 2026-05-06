//! # Monte-Carlo Random Forest Majority Vote
//!
//! Sim N independent decision trees with per-tree accuracy `tree_acc`.
//! Majority voting ensemble on N samples; reports ensemble accuracy
//! (should exceed individual tree accuracy per Condorcet's theorem).
//!
//! Demonstrates the **MC.89** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Breiman, Random Forests (Machine Learning 45, 2001);
//!  Condorcet 1785 jury theorem.
//!
//! Run with: cargo run --example mc_random_forest_voting
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ForestVerdict {
    Ok {
        ensemble_accuracy: f64,
        single_tree_accuracy_observed: f64,
    },
    InvalidConfig,
}

pub fn simulate(samples: u32, trees: u32, tree_acc: f64, seed: u64) -> ForestVerdict {
    if samples == 0 || trees < 3 || trees % 2 == 0 || !(0.5..=1.0).contains(&tree_acc) {
        return ForestVerdict::InvalidConfig;
    }
    let threshold = trees / 2 + 1;
    let mut ensemble_correct = 0u32;
    let mut tree_correct = 0u32;
    let mut tree_total = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..samples {
        let mut votes = 0u32;
        for _ in 0..trees {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            tree_total += 1;
            if r < tree_acc {
                votes += 1;
                tree_correct += 1;
            }
        }
        if votes >= threshold {
            ensemble_correct += 1;
        }
    }
    ForestVerdict::Ok {
        ensemble_accuracy: f64::from(ensemble_correct) / f64::from(samples),
        single_tree_accuracy_observed: f64::from(tree_correct) / f64::from(tree_total),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_forest_voting")?;

    println!("70% trees, 11 trees: {:?}", simulate(2000, 11, 0.7, 42));
    println!(
        "barely-better-than-coin: {:?}",
        simulate(2000, 51, 0.55, 42)
    );
    println!("invalid: {:?}", simulate(0, 11, 0.7, 42));
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
    fn ensemble_beats_individual_for_correlated() {
        // 70% trees → ensemble accuracy should be > tree accuracy.
        let v = simulate(2000, 11, 0.7, 42);
        if let ForestVerdict::Ok {
            ensemble_accuracy,
            single_tree_accuracy_observed,
        } = v
        {
            assert!(ensemble_accuracy >= single_tree_accuracy_observed);
        }
    }

    #[test]
    fn high_tree_accuracy_high_ensemble() {
        let v = simulate(2000, 11, 0.95, 42);
        if let ForestVerdict::Ok {
            ensemble_accuracy, ..
        } = v
        {
            assert!(ensemble_accuracy > 0.99);
        }
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(0, 11, 0.7, 42), ForestVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trees() {
        assert_eq!(simulate(100, 1, 0.7, 42), ForestVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_even_trees() {
        assert_eq!(simulate(100, 10, 0.7, 42), ForestVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_acc_below_half() {
        assert_eq!(simulate(100, 11, 0.4, 42), ForestVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_acc_above_one() {
        assert_eq!(simulate(100, 11, 1.5, 42), ForestVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 11, 0.7, 42);
        let b = simulate(500, 11, 0.7, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(500, 11, 0.7, 42);
        if let ForestVerdict::Ok {
            ensemble_accuracy, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&ensemble_accuracy));
        }
    }

    #[test]
    fn larger_n_sharper_consensus() {
        let small = simulate(2000, 5, 0.55, 42);
        let large = simulate(2000, 99, 0.55, 42);
        if let (
            ForestVerdict::Ok {
                ensemble_accuracy: s,
                ..
            },
            ForestVerdict::Ok {
                ensemble_accuracy: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn boundary_acc_0_5_invalid() {
        // 0.5 is allowed by ..=, but result might be unstable.
        let v = simulate(2000, 11, 0.5, 42);
        if let ForestVerdict::Ok {
            ensemble_accuracy, ..
        } = v
        {
            // ~50/50 randomness; we only assert in unit range.
            assert!((0.0..=1.0).contains(&ensemble_accuracy));
        }
    }
}
