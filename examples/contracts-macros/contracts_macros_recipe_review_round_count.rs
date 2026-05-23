//! # Contracts-Macros Recipe Review Round Count
//!
//! Track per-recipe code review round counts. Flag recipes
//! exceeding `max_rounds` (sign of design churn).
//!
//! Demonstrates the **CMM.108** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bird & Bacchelli, "Expectations Outcomes and Challenges
//!  of Modern Code Review" (ICSE 2013); GitHub PR review-round
//!  diagnostic conventions.
//!
//! Run with: cargo run --example contracts_macros_recipe_review_round_count
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReviewVerdict {
    Ok {
        excessive: Vec<String>,
        avg_rounds: f64,
        max_rounds_observed: u32,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, u32)], max_rounds: u32) -> ReviewVerdict {
    if recipes.is_empty() || max_rounds == 0 {
        return ReviewVerdict::InvalidConfig;
    }
    let mut excessive: Vec<String> = Vec::new();
    let mut total_rounds: u64 = 0;
    let mut max_obs: u32 = 0;
    for (name, rounds) in recipes {
        if *rounds > max_rounds {
            excessive.push((*name).to_string());
        }
        total_rounds += u64::from(*rounds);
        if *rounds > max_obs {
            max_obs = *rounds;
        }
    }
    excessive.sort();
    let avg_rounds = total_rounds as f64 / recipes.len() as f64;
    ReviewVerdict::Ok {
        excessive,
        avg_rounds,
        max_rounds_observed: max_obs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_review_round_count")?;

    let recipes = [("r1", 2), ("r2", 8), ("r3", 1), ("r4", 12)];
    println!("audit max=3: {:?}", audit(&recipes, 3));
    println!("invalid: {:?}", audit(&[], 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_threshold_no_offender() {
        let recipes = [("r", 1)];
        let v = audit(&recipes, 3);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert!(excessive.is_empty());
        }
    }

    #[test]
    fn over_threshold_flagged() {
        let recipes = [("r", 5)];
        let v = audit(&recipes, 3);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert_eq!(excessive, vec!["r".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 3), ReviewVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        let recipes = [("r", 5)];
        assert_eq!(audit(&recipes, 0), ReviewVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_max_no_flag() {
        let recipes = [("r", 3)];
        let v = audit(&recipes, 3);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert!(excessive.is_empty());
        }
    }

    #[test]
    fn one_over_max_flagged() {
        let recipes = [("r", 4)];
        let v = audit(&recipes, 3);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert_eq!(excessive, vec!["r".to_string()]);
        }
    }

    #[test]
    fn average_correct() {
        let recipes = [("a", 2), ("b", 4)];
        let v = audit(&recipes, 10);
        if let ReviewVerdict::Ok { avg_rounds, .. } = v {
            assert!((avg_rounds - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn max_observed_correct() {
        let recipes = [("a", 2), ("b", 7), ("c", 4)];
        let v = audit(&recipes, 10);
        if let ReviewVerdict::Ok {
            max_rounds_observed,
            ..
        } = v
        {
            assert_eq!(max_rounds_observed, 7);
        }
    }

    #[test]
    fn excessive_sorted() {
        let recipes = [("zeta", 10), ("alpha", 10)];
        let v = audit(&recipes, 5);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert_eq!(excessive, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let recipes = [("r", 2)];
        let r1 = audit(&recipes, 3);
        let r2 = audit(&recipes, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32)> = (0..30).map(|_| ("r", 2)).collect();
        let v = audit(&recipes, 3);
        if let ReviewVerdict::Ok { excessive, .. } = v {
            assert!(excessive.is_empty());
        }
    }
}
