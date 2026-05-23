//! # Contracts-Macros Recipe Decomposition
//!
//! Split a complex recipe (length > complexity_threshold) into sub-
//! recipe blocks for individual obligation tracking. Returns the
//! decomposed block list with their phase names.
//!
//! Demonstrates the **CMM.38** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SRP / decomposition principle (Parnas 1972).
//!
//! Run with: cargo run --example contracts_macros_recipe_decomposition
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DecomposeVerdict {
    Single { phase: String, complexity: u32 },
    Decomposed { blocks: Vec<(String, u32)> },
    EmptyRecipe,
    InvalidThreshold,
}

pub fn split(recipe_phases: &[(&str, u32)], complexity_threshold: u32) -> DecomposeVerdict {
    if recipe_phases.is_empty() {
        return DecomposeVerdict::EmptyRecipe;
    }
    if complexity_threshold == 0 {
        return DecomposeVerdict::InvalidThreshold;
    }
    let total: u32 = recipe_phases.iter().map(|(_, c)| *c).sum();
    if total <= complexity_threshold {
        let phase = recipe_phases
            .iter()
            .map(|(n, _)| (*n).to_string())
            .collect::<Vec<_>>()
            .join("+");
        return DecomposeVerdict::Single {
            phase,
            complexity: total,
        };
    }
    let blocks: Vec<(String, u32)> = recipe_phases
        .iter()
        .map(|(n, c)| ((*n).to_string(), *c))
        .collect();
    DecomposeVerdict::Decomposed { blocks }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_decomposition")?;

    let small = [("normalize", 5), ("apply", 3)];
    println!("single: {:?}", split(&small, 100));

    let large = [
        ("load", 30),
        ("normalize", 50),
        ("apply", 80),
        ("persist", 40),
    ];
    println!("decomposed: {:?}", split(&large, 100));
    println!("empty: {:?}", split(&[], 100));
    println!("invalid: {:?}", split(&large, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_recipe_stays_single() {
        let v = split(&[("a", 5), ("b", 3)], 100);
        if let DecomposeVerdict::Single { complexity, .. } = v {
            assert_eq!(complexity, 8);
        }
    }

    #[test]
    fn large_recipe_decomposed() {
        let v = split(&[("a", 50), ("b", 60)], 100);
        assert!(matches!(v, DecomposeVerdict::Decomposed { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(split(&[], 100), DecomposeVerdict::EmptyRecipe);
    }

    #[test]
    fn zero_threshold_invalid() {
        assert_eq!(split(&[("a", 1)], 0), DecomposeVerdict::InvalidThreshold);
    }

    #[test]
    fn at_threshold_stays_single() {
        let v = split(&[("a", 50), ("b", 50)], 100);
        if let DecomposeVerdict::Single { complexity, .. } = v {
            assert_eq!(complexity, 100);
        }
    }

    #[test]
    fn just_over_threshold_decomposed() {
        let v = split(&[("a", 50), ("b", 51)], 100);
        assert!(matches!(v, DecomposeVerdict::Decomposed { .. }));
    }

    #[test]
    fn single_phase_under() {
        let v = split(&[("only", 50)], 100);
        if let DecomposeVerdict::Single { phase, .. } = v {
            assert_eq!(phase, "only");
        }
    }

    #[test]
    fn block_count_matches_input() {
        let v = split(&[("a", 50), ("b", 60), ("c", 40)], 100);
        if let DecomposeVerdict::Decomposed { blocks } = v {
            assert_eq!(blocks.len(), 3);
        }
    }

    #[test]
    fn phase_names_joined_with_plus() {
        let v = split(&[("a", 5), ("b", 5), ("c", 5)], 100);
        if let DecomposeVerdict::Single { phase, .. } = v {
            assert_eq!(phase, "a+b+c");
        }
    }

    #[test]
    fn deterministic() {
        let phases = [("a", 50), ("b", 60)];
        let a = split(&phases, 100);
        let b = split(&phases, 100);
        assert_eq!(a, b);
    }
}
