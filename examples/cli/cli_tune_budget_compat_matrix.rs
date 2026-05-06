//! # apr tune — Budget × Strategy Compatibility Matrix
//!
//! `apr tune --strategy <S> --trials <N>` compatibility: grid is
//! exhaustive (N must equal product-of-cardinalities); random/TPE accept
//! any N. This recipe builds the cardinality validator and the
//! compatibility matrix.
//!
//! Demonstrates the **TUNE.9** recipe for PMAT-111 (apr tune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HPO-001
//!
//! Run with: cargo run --example cli_tune_budget_compat_matrix
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Grid,
    Random,
    Tpe,
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok,
    GridUnderBudget { needed: u64, given: u64 },
    GridOverBudget { needed: u64, given: u64 },
    ZeroTrials,
}

pub fn validate_budget(
    strategy: Strategy,
    num_trials: u64,
    grid_cardinalities: &[u64],
) -> BudgetVerdict {
    if num_trials == 0 {
        return BudgetVerdict::ZeroTrials;
    }
    if strategy != Strategy::Grid {
        return BudgetVerdict::Ok;
    }
    let product: u64 = grid_cardinalities.iter().product();
    if product == 0 {
        // Any zero-cardinality dim → empty space.
        return BudgetVerdict::ZeroTrials;
    }
    match num_trials.cmp(&product) {
        std::cmp::Ordering::Less => BudgetVerdict::GridUnderBudget {
            needed: product,
            given: num_trials,
        },
        std::cmp::Ordering::Greater => BudgetVerdict::GridOverBudget {
            needed: product,
            given: num_trials,
        },
        std::cmp::Ordering::Equal => BudgetVerdict::Ok,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tune_budget_compat_matrix")?;

    let cases = [
        ("random/100/[]", Strategy::Random, 100, vec![]),
        ("grid/12/[3,4]", Strategy::Grid, 12, vec![3, 4]),
        ("grid/10/[3,4]", Strategy::Grid, 10, vec![3, 4]),
        ("grid/100/[3,4]", Strategy::Grid, 100, vec![3, 4]),
        ("zero", Strategy::Tpe, 0, vec![]),
    ];
    for (label, s, n, c) in cases {
        println!("{label:>20}  →  {:?}", validate_budget(s, n, &c));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn random_any_budget_ok() {
        // Random doesn't care about cardinality.
        assert_eq!(
            validate_budget(Strategy::Random, 100, &[]),
            BudgetVerdict::Ok
        );
    }

    #[test]
    fn tpe_any_budget_ok() {
        assert_eq!(validate_budget(Strategy::Tpe, 50, &[]), BudgetVerdict::Ok);
    }

    #[test]
    fn grid_exact_match_ok() {
        // 3 × 4 = 12 trials needed.
        assert_eq!(
            validate_budget(Strategy::Grid, 12, &[3, 4]),
            BudgetVerdict::Ok
        );
    }

    #[test]
    fn grid_under_budget_rejected() {
        let v = validate_budget(Strategy::Grid, 10, &[3, 4]);
        assert!(matches!(
            v,
            BudgetVerdict::GridUnderBudget {
                needed: 12,
                given: 10
            }
        ));
    }

    #[test]
    fn grid_over_budget_rejected() {
        let v = validate_budget(Strategy::Grid, 100, &[3, 4]);
        assert!(matches!(
            v,
            BudgetVerdict::GridOverBudget {
                needed: 12,
                given: 100
            }
        ));
    }

    #[test]
    fn zero_trials_rejected() {
        assert_eq!(
            validate_budget(Strategy::Random, 0, &[]),
            BudgetVerdict::ZeroTrials
        );
    }

    #[test]
    fn zero_cardinality_dim_rejected() {
        // 3 × 0 = 0 → empty space, no valid trials.
        assert_eq!(
            validate_budget(Strategy::Grid, 5, &[3, 0]),
            BudgetVerdict::ZeroTrials
        );
    }

    #[test]
    fn grid_three_dim_cardinality() {
        // 2 × 3 × 5 = 30 needed.
        assert_eq!(
            validate_budget(Strategy::Grid, 30, &[2, 3, 5]),
            BudgetVerdict::Ok
        );
    }
}
