//! # Analysis — `pv check-parity` Authoring Pattern
//!
//! `pv check-parity` is a SEMANTIC gate (FALSIFY-CODE-PARITY-001..005) for
//! parity-matrix contracts (apr-code-parity-v1.yaml is the canonical
//! example). This recipe demonstrates authoring a small parity matrix with
//! `cross_check_command` per row and the gate that runs each one and
//! asserts `expected_min_hits` / `expected_max_hits` bounds.
//!
//! Demonstrates the **AN+.3** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml v5.1 (the canonical parity matrix) + Stol & Avgeriou (2010). Patterns for Variability Management. SoSyM 9(4)
//!
//! Run with: cargo run --example analysis_pv_check_parity_authoring
//!
//! Added by PMAT-086 (expand-cookbooks: Tier 4 authoring patterns).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug)]
struct ParityRow {
    id: String,
    cross_check_value: usize, // simulated: result of running cross_check_command
    expected_min_hits: usize,
    expected_max_hits: Option<usize>,
}

#[derive(Debug, PartialEq)]
enum CheckOutcome {
    Pass,
    BelowMin,
    AboveMax,
}

fn check_row(row: &ParityRow) -> CheckOutcome {
    if row.cross_check_value < row.expected_min_hits {
        return CheckOutcome::BelowMin;
    }
    if let Some(max) = row.expected_max_hits {
        if row.cross_check_value > max {
            return CheckOutcome::AboveMax;
        }
    }
    CheckOutcome::Pass
}

fn run_parity_check(rows: &[ParityRow]) -> Result<()> {
    let mut failed = Vec::new();
    for row in rows {
        let outcome = check_row(row);
        if outcome != CheckOutcome::Pass {
            failed.push(format!("{}: {:?}", row.id, outcome));
        }
    }
    if !failed.is_empty() {
        return Err(apr_cookbook::CookbookError::Validation(format!(
            "FALSIFY-CODE-PARITY-001: {} parity rows failed:\n  {}",
            failed.len(),
            failed.join("\n  ")
        )));
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_pv_check_parity_authoring")?;
    let rows = [
        ParityRow {
            id: "mcp-client".into(),
            cross_check_value: 4, // grep -c "fn parse_mcp_*" returned 4
            expected_min_hits: 4,
            expected_max_hits: None,
        },
        ParityRow {
            id: "custom-agents".into(),
            cross_check_value: 4, // grep -c "fn (parse_agent_md|...)" returned 4
            expected_min_hits: 4,
            expected_max_hits: None,
        },
        ParityRow {
            id: "session-management".into(),
            cross_check_value: 2, // grep -c "fn session_*" returned 2 — below min 3
            expected_min_hits: 3,
            expected_max_hits: None,
        },
    ];

    match run_parity_check(&rows) {
        Ok(()) => println!("all rows pass"),
        Err(e) => println!("{e}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pass_when_within_bounds() {
        let row = ParityRow {
            id: "x".into(),
            cross_check_value: 5,
            expected_min_hits: 3,
            expected_max_hits: Some(10),
        };
        assert_eq!(check_row(&row), CheckOutcome::Pass);
    }

    #[test]
    fn below_min_fails() {
        let row = ParityRow {
            id: "x".into(),
            cross_check_value: 2,
            expected_min_hits: 3,
            expected_max_hits: None,
        };
        assert_eq!(check_row(&row), CheckOutcome::BelowMin);
    }

    #[test]
    fn above_max_fails() {
        let row = ParityRow {
            id: "x".into(),
            cross_check_value: 11,
            expected_min_hits: 3,
            expected_max_hits: Some(10),
        };
        assert_eq!(check_row(&row), CheckOutcome::AboveMax);
    }
}
