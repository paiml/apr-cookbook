//! # apr qa — Warmup + Iteration Budget
//!
//! `apr qa <FILE> --warmup <W> --iterations <N> --max-tokens <T>` controls
//! the throughput-measurement budget. The total tokens generated is
//! `(W + N) * T`. This recipe exposes the budget computation so a CI
//! pipeline can predict wall-clock cost before the run, and asserts the
//! IIUR-required floor (warmup ≥ 1, iterations ≥ 1, max_tokens ≥ 1) so
//! degenerate budgets are rejected at the boundary.
//!
//! Demonstrates the **QA.4** recipe for PMAT-093 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender F-PERF-042 + warm-pass measurement convention
//!
//! Run with: cargo run --example cli_qa_warmup_iteration_budget
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QaBudget {
    pub warmup: u32,
    pub iterations: u32,
    pub max_tokens: u32,
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok {
        total_tokens: u64,
        measured_tokens: u64,
    },
    InvalidWarmup,
    InvalidIterations,
    InvalidMaxTokens,
}

pub fn validate_budget(b: QaBudget) -> BudgetVerdict {
    if b.warmup == 0 {
        return BudgetVerdict::InvalidWarmup;
    }
    if b.iterations == 0 {
        return BudgetVerdict::InvalidIterations;
    }
    if b.max_tokens == 0 {
        return BudgetVerdict::InvalidMaxTokens;
    }
    let total = (u64::from(b.warmup) + u64::from(b.iterations)) * u64::from(b.max_tokens);
    let measured = u64::from(b.iterations) * u64::from(b.max_tokens);
    BudgetVerdict::Ok {
        total_tokens: total,
        measured_tokens: measured,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_warmup_iteration_budget")?;

    let cases = [
        (
            "default",
            QaBudget {
                warmup: 3,
                iterations: 10,
                max_tokens: 32,
            },
        ),
        (
            "smoke",
            QaBudget {
                warmup: 1,
                iterations: 1,
                max_tokens: 8,
            },
        ),
        (
            "long-tail",
            QaBudget {
                warmup: 5,
                iterations: 50,
                max_tokens: 512,
            },
        ),
        (
            "zero warm",
            QaBudget {
                warmup: 0,
                iterations: 10,
                max_tokens: 32,
            },
        ),
        (
            "zero tok",
            QaBudget {
                warmup: 1,
                iterations: 1,
                max_tokens: 0,
            },
        ),
    ];

    println!("=== Recipe: cli_qa_warmup_iteration_budget ===");
    for (label, b) in cases {
        println!(
            "{label:>10}  warm={:>2} iter={:>2} tok={:>3}  →  {:?}",
            b.warmup,
            b.iterations,
            b.max_tokens,
            validate_budget(b)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_budget_computes_total_and_measured() {
        let b = QaBudget {
            warmup: 3,
            iterations: 10,
            max_tokens: 32,
        };
        if let BudgetVerdict::Ok {
            total_tokens,
            measured_tokens,
        } = validate_budget(b)
        {
            assert_eq!(total_tokens, 13 * 32);
            assert_eq!(measured_tokens, 10 * 32);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn measured_excludes_warmup() {
        // Warmup tokens ARE generated (cost wall-clock) but not measured.
        // measured_tokens must equal iterations * max_tokens regardless of warmup.
        let b = QaBudget {
            warmup: 100,
            iterations: 5,
            max_tokens: 16,
        };
        if let BudgetVerdict::Ok {
            measured_tokens, ..
        } = validate_budget(b)
        {
            assert_eq!(measured_tokens, 80);
        }
    }

    #[test]
    fn zero_warmup_rejected() {
        // Without warmup the measurement includes JIT/cache-cold effects.
        let b = QaBudget {
            warmup: 0,
            iterations: 10,
            max_tokens: 32,
        };
        assert_eq!(validate_budget(b), BudgetVerdict::InvalidWarmup);
    }

    #[test]
    fn zero_iterations_rejected() {
        let b = QaBudget {
            warmup: 1,
            iterations: 0,
            max_tokens: 32,
        };
        assert_eq!(validate_budget(b), BudgetVerdict::InvalidIterations);
    }

    #[test]
    fn zero_max_tokens_rejected() {
        let b = QaBudget {
            warmup: 1,
            iterations: 1,
            max_tokens: 0,
        };
        assert_eq!(validate_budget(b), BudgetVerdict::InvalidMaxTokens);
    }

    #[test]
    fn large_budget_does_not_overflow() {
        // u32 * u32 = up to 2^64; pinned via u64::from in the impl.
        let b = QaBudget {
            warmup: u32::MAX,
            iterations: 1,
            max_tokens: u32::MAX,
        };
        // This should compute without panic.
        let v = validate_budget(b);
        assert!(matches!(v, BudgetVerdict::Ok { .. }));
    }
}
