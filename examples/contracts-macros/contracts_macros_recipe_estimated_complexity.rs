//! # Contracts-Macros Recipe Estimated Complexity
//!
//! Estimate complexity score from `(lines, tests)` pair using
//! formula `complexity = lines / max(tests, 1) * 10`. Returns score
//! per recipe, mean, and high-complexity flag list.
//!
//! Demonstrates the **CMM.126** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: McCabe cyclomatic complexity (1976); test-density
//!  metrics in software engineering.
//!
//! Run with: cargo run --example contracts_macros_recipe_estimated_complexity
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ComplexityVerdict {
    Ok {
        per_recipe: Vec<(String, u32)>,
        mean_score: f64,
        high_complexity: Vec<String>,
    },
    InvalidConfig,
}

pub fn estimate(recipes: &[(&str, u32, u32)], high_threshold: u32) -> ComplexityVerdict {
    if recipes.is_empty() || high_threshold == 0 {
        return ComplexityVerdict::InvalidConfig;
    }
    let mut per_recipe: Vec<(String, u32)> = Vec::with_capacity(recipes.len());
    let mut high_complexity: Vec<String> = Vec::new();
    let mut total: u64 = 0;
    for (name, lines, tests) in recipes {
        let score = lines * 10 / (*tests).max(1);
        per_recipe.push(((*name).to_string(), score));
        total += u64::from(score);
        if score > high_threshold {
            high_complexity.push((*name).to_string());
        }
    }
    high_complexity.sort();
    let mean_score = total as f64 / recipes.len() as f64;
    ComplexityVerdict::Ok {
        per_recipe,
        mean_score,
        high_complexity,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_estimated_complexity")?;

    let recipes = [("simple", 100, 10), ("complex", 500, 5), ("ok", 200, 20)];
    println!("audit threshold=200: {:?}", estimate(&recipes, 200));
    println!("invalid: {:?}", estimate(&[], 200));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn score_correct() {
        let recipes = [("a", 100, 10)];
        let v = estimate(&recipes, 1000);
        if let ComplexityVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, 100); // 100 * 10 / 10 = 100
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(estimate(&[], 200), ComplexityVerdict::InvalidConfig);
    }

    #[test]
    fn zero_threshold_rejected() {
        let recipes = [("a", 100, 10)];
        assert_eq!(estimate(&recipes, 0), ComplexityVerdict::InvalidConfig);
    }

    #[test]
    fn zero_tests_treated_as_one() {
        let recipes = [("a", 100, 0)];
        let v = estimate(&recipes, 10000);
        if let ComplexityVerdict::Ok { per_recipe, .. } = v {
            // 100 * 10 / max(0, 1) = 100 * 10 / 1 = 1000.
            assert_eq!(per_recipe[0].1, 1000);
        }
    }

    #[test]
    fn high_complexity_flagged() {
        let recipes = [("a", 1000, 10)];
        let v = estimate(&recipes, 500);
        if let ComplexityVerdict::Ok {
            high_complexity, ..
        } = v
        {
            assert_eq!(high_complexity, vec!["a".to_string()]);
        }
    }

    #[test]
    fn low_complexity_not_flagged() {
        let recipes = [("a", 50, 10)];
        let v = estimate(&recipes, 200);
        if let ComplexityVerdict::Ok {
            high_complexity, ..
        } = v
        {
            assert!(high_complexity.is_empty());
        }
    }

    #[test]
    fn mean_score_correct() {
        let recipes = [("a", 100, 10), ("b", 200, 10)];
        let v = estimate(&recipes, 1000);
        if let ComplexityVerdict::Ok { mean_score, .. } = v {
            // Scores: 100, 200 → mean 150.
            assert!((mean_score - 150.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let recipes = [("a", 100, 10)];
        let r1 = estimate(&recipes, 200);
        let r2 = estimate(&recipes, 200);
        assert_eq!(r1, r2);
    }

    #[test]
    fn high_complexity_sorted() {
        let recipes = [("zeta", 1000, 10), ("alpha", 1000, 10)];
        let v = estimate(&recipes, 100);
        if let ComplexityVerdict::Ok {
            high_complexity, ..
        } = v
        {
            assert_eq!(
                high_complexity,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32, u32)> = (0..50).map(|_| ("r", 50, 10)).collect();
        let v = estimate(&recipes, 1000);
        if let ComplexityVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe.len(), 50);
        }
    }

    #[test]
    fn many_tests_lower_score() {
        let r1 = [("a", 100, 5)];
        let r2 = [("a", 100, 50)];
        let v1 = estimate(&r1, 1000);
        let v2 = estimate(&r2, 1000);
        if let (
            ComplexityVerdict::Ok { per_recipe: p1, .. },
            ComplexityVerdict::Ok { per_recipe: p2, .. },
        ) = (v1, v2)
        {
            assert!(p1[0].1 > p2[0].1);
        }
    }
}
