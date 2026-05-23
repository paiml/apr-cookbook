//! # WASM Import Count Budget
//!
//! Validate the WASM module's import section against a count budget
//! per kind (function, memory, table, global). Returns over-budget
//! kinds.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §5.5.5 import section; wasm-validate
//!  import-budget rules.
//!
//! Run with: cargo run --example wasm_import_count_budget
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ImportBudgetVerdict {
    Ok {
        over_budget_kinds: Vec<String>,
        total_imports: u32,
    },
    InvalidConfig,
}

/// Counts: (functions, memories, tables, globals).
/// Budgets: (max_funcs, max_memories, max_tables, max_globals).
pub fn check(counts: (u32, u32, u32, u32), budgets: (u32, u32, u32, u32)) -> ImportBudgetVerdict {
    let (cf, cm, ct, cg) = counts;
    let (bf, bm, bt, bg) = budgets;
    if bf == 0 && bm == 0 && bt == 0 && bg == 0 {
        return ImportBudgetVerdict::InvalidConfig;
    }
    let mut over: Vec<String> = Vec::new();
    if cf > bf {
        over.push("function".to_string());
    }
    if cm > bm {
        over.push("memory".to_string());
    }
    if ct > bt {
        over.push("table".to_string());
    }
    if cg > bg {
        over.push("global".to_string());
    }
    over.sort();
    let total = cf + cm + ct + cg;
    ImportBudgetVerdict::Ok {
        over_budget_kinds: over,
        total_imports: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_import_count_budget")?;

    println!("ok: {:?}", check((10, 1, 1, 5), (50, 1, 1, 10)));
    println!("over: {:?}", check((100, 1, 1, 5), (50, 1, 1, 10)));
    println!("invalid: {:?}", check((10, 1, 1, 5), (0, 0, 0, 0)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_budget_no_violations() {
        let v = check((10, 1, 1, 5), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert!(over_budget_kinds.is_empty());
        }
    }

    #[test]
    fn function_over_budget() {
        let v = check((100, 0, 0, 0), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert_eq!(over_budget_kinds, vec!["function".to_string()]);
        }
    }

    #[test]
    fn memory_over_budget() {
        let v = check((0, 5, 0, 0), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert_eq!(over_budget_kinds, vec!["memory".to_string()]);
        }
    }

    #[test]
    fn all_budgets_zero_rejected() {
        assert_eq!(
            check((0, 0, 0, 0), (0, 0, 0, 0)),
            ImportBudgetVerdict::InvalidConfig
        );
    }

    #[test]
    fn at_budget_no_violation() {
        let v = check((50, 1, 1, 10), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert!(over_budget_kinds.is_empty());
        }
    }

    #[test]
    fn total_imports_correct() {
        let v = check((10, 1, 1, 5), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok { total_imports, .. } = v {
            assert_eq!(total_imports, 17);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check((10, 1, 1, 5), (50, 1, 1, 10));
        let r2 = check((10, 1, 1, 5), (50, 1, 1, 10));
        assert_eq!(r1, r2);
    }

    #[test]
    fn multiple_kinds_over() {
        let v = check((100, 5, 5, 100), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert_eq!(over_budget_kinds.len(), 4);
        }
    }

    #[test]
    fn over_kinds_sorted() {
        // Order depends on which fields are over-budget.
        let v = check((100, 5, 0, 0), (50, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds, ..
        } = v
        {
            assert_eq!(over_budget_kinds, vec!["function", "memory"]);
        }
    }

    #[test]
    fn high_counts_handled() {
        let v = check((1000, 1, 1, 100), (10000, 10, 10, 1000));
        assert!(matches!(v, ImportBudgetVerdict::Ok { .. }));
    }

    #[test]
    fn zero_counts_no_violations() {
        let v = check((0, 0, 0, 0), (10, 1, 1, 10));
        if let ImportBudgetVerdict::Ok {
            over_budget_kinds,
            total_imports,
        } = v
        {
            assert!(over_budget_kinds.is_empty());
            assert_eq!(total_imports, 0);
        }
    }
}
