//! # apr parity --assert — Exit Code Mapper
//!
//! `apr parity <FILE> --assert` exits non-zero on any divergence (CI mode).
//! Without `--assert`, divergence is informational. This recipe documents
//! the exit-code contract: 0 = parity, 65 = divergence under --assert
//! (EX_DATAERR), 70 = internal error (EX_SOFTWARE).
//!
//! Demonstrates the **PARITY.10** recipe for PMAT-111 (apr parity coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-232 + sysexits.h conventions
//!
//! Run with: cargo run --example cli_parity_assert_mode_exit_codes
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub enum ParityOutcome {
    Match,
    Diverged,
    InternalError,
}

const EX_OK: i32 = 0;
const EX_DATAERR: i32 = 65;
const EX_SOFTWARE: i32 = 70;

pub fn exit_code(outcome: ParityOutcome, assert_mode: bool) -> i32 {
    match (outcome, assert_mode) {
        (ParityOutcome::Match, _) => EX_OK,
        (ParityOutcome::Diverged, true) => EX_DATAERR,
        (ParityOutcome::Diverged, false) => EX_OK,
        (ParityOutcome::InternalError, _) => EX_SOFTWARE,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_parity_assert_mode_exit_codes")?;

    let cases = [
        ("match no-assert", ParityOutcome::Match, false),
        ("match assert", ParityOutcome::Match, true),
        ("diverged no-assert", ParityOutcome::Diverged, false),
        ("diverged assert", ParityOutcome::Diverged, true),
        ("error no-assert", ParityOutcome::InternalError, false),
        ("error assert", ParityOutcome::InternalError, true),
    ];
    for (label, outcome, assert) in cases {
        println!("{label:>22}  →  exit {}", exit_code(outcome, assert));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn match_always_zero() {
        assert_eq!(exit_code(ParityOutcome::Match, true), 0);
        assert_eq!(exit_code(ParityOutcome::Match, false), 0);
    }

    #[test]
    fn diverged_no_assert_zero() {
        // Without --assert, divergence is informational only.
        assert_eq!(exit_code(ParityOutcome::Diverged, false), 0);
    }

    #[test]
    fn diverged_assert_returns_ex_dataerr() {
        // With --assert, divergence triggers EX_DATAERR (65).
        assert_eq!(exit_code(ParityOutcome::Diverged, true), EX_DATAERR);
    }

    #[test]
    fn internal_error_always_returns_ex_software() {
        // EX_SOFTWARE (70) regardless of --assert flag.
        assert_eq!(exit_code(ParityOutcome::InternalError, true), EX_SOFTWARE);
        assert_eq!(exit_code(ParityOutcome::InternalError, false), EX_SOFTWARE);
    }

    #[test]
    fn dataerr_and_software_are_distinct() {
        assert_ne!(EX_DATAERR, EX_SOFTWARE);
    }

    #[test]
    fn ex_dataerr_is_65_per_sysexits() {
        // sysexits.h: EX_DATAERR = 65 (data format error).
        assert_eq!(EX_DATAERR, 65);
    }

    #[test]
    fn ex_software_is_70_per_sysexits() {
        // sysexits.h: EX_SOFTWARE = 70 (internal software error).
        assert_eq!(EX_SOFTWARE, 70);
    }
}
