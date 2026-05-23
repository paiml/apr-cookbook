//! # apr encrypt — `--force` Overwrite Policy
//!
//! `apr encrypt --output <FILE>` defaults to refusing to overwrite an
//! existing output file. `--force` opts in. This recipe builds the
//! decision tree and asserts the contract: missing output → write,
//! existing output without --force → refuse, existing with --force →
//! overwrite (and emit a warning that it happened).
//!
//! Demonstrates the **ENCRYPT.6** recipe for PMAT-103 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + safe-by-default convention
//!
//! Run with: cargo run --example cli_encrypt_force_overwrite_policy
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WriteAction {
    Create,
    Overwrite { warning: &'static str },
    Refused { reason: &'static str },
}

pub fn decide_write(output_exists: bool, force: bool, output_path: &str) -> WriteAction {
    if output_path.is_empty() {
        return WriteAction::Refused {
            reason: "output path is empty",
        };
    }
    match (output_exists, force) {
        (false, _) => WriteAction::Create,
        (true, false) => WriteAction::Refused {
            reason: "output file exists; pass --force to overwrite",
        },
        (true, true) => WriteAction::Overwrite {
            warning: "overwriting existing file (--force)",
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_force_overwrite_policy")?;

    let cases = [
        ("missing-no-force", "out.enc", false, false),
        ("missing-force", "out.enc", false, true),
        ("exists-no-force", "out.enc", true, false),
        ("exists-force", "out.enc", true, true),
        ("empty-path", "", false, false),
    ];

    for (label, path, exists, force) in cases {
        println!("{label:>20}  →  {:?}", decide_write(exists, force, path));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_output_creates_without_force() {
        assert_eq!(decide_write(false, false, "out.enc"), WriteAction::Create);
    }

    #[test]
    fn missing_output_creates_with_force() {
        // --force on a missing file is harmless — still Create.
        assert_eq!(decide_write(false, true, "out.enc"), WriteAction::Create);
    }

    #[test]
    fn existing_output_refused_without_force() {
        let v = decide_write(true, false, "out.enc");
        assert!(matches!(v, WriteAction::Refused { .. }));
    }

    #[test]
    fn existing_output_overwritten_with_force() {
        let v = decide_write(true, true, "out.enc");
        assert!(matches!(v, WriteAction::Overwrite { .. }));
    }

    #[test]
    fn empty_path_always_refused() {
        // Empty path is invalid regardless of --force or existence.
        for exists in [false, true] {
            for force in [false, true] {
                let v = decide_write(exists, force, "");
                assert!(matches!(v, WriteAction::Refused { .. }));
            }
        }
    }

    #[test]
    fn overwrite_includes_warning_string() {
        // The warning is what gets logged to stderr — verify it's non-empty.
        if let WriteAction::Overwrite { warning } = decide_write(true, true, "x") {
            assert!(!warning.is_empty());
        } else {
            panic!("expected Overwrite");
        }
    }
}
