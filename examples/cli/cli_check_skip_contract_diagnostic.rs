//! # apr check — `--skip-contract` Diagnostic Mode
//!
//! `apr check --skip-contract <MODEL.apr>` runs the integrity pipeline but
//! omits the (slow) contract round-trip stage. This is the diagnostic mode
//! used during model migration when the attached contract is intentionally
//! out-of-date and re-stamping would discard valuable failure signal from
//! the other 9 stages.
//!
//! The recipe asserts that `--skip-contract` produces stage 10 = Skipped,
//! never quietly Pass — silent skipping would let bad contracts ship.
//!
//! Demonstrates the **CHECK.2** recipe for PMAT-088 (apr check coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CHECK-002 + provable-contracts v0.31.x
//!
//! Run with: cargo run --example cli_check_skip_contract_diagnostic
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
enum StageStatus {
    Pass,
    #[allow(dead_code)]
    // present for API parity with the broader CHECK pipeline; not exercised in this recipe
    Fail(&'static str),
    Skipped(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
struct StageOutcome {
    name: &'static str,
    status: StageStatus,
}

#[derive(Debug, Default)]
struct CheckFlags {
    skip_contract: bool,
    skip_signature: bool,
}

fn check(flags: &CheckFlags) -> Vec<StageOutcome> {
    [
        ("magic", StageStatus::Pass),
        ("version", StageStatus::Pass),
        ("crc32", StageStatus::Pass),
        ("tensor-shape", StageStatus::Pass),
        ("tensor-dtype", StageStatus::Pass),
        ("quantization", StageStatus::Pass),
        ("tokenizer", StageStatus::Pass),
        ("provenance", StageStatus::Pass),
        (
            "signature",
            if flags.skip_signature {
                StageStatus::Skipped("--skip-signature")
            } else {
                StageStatus::Pass
            },
        ),
        (
            "contract",
            if flags.skip_contract {
                StageStatus::Skipped("--skip-contract")
            } else {
                StageStatus::Pass
            },
        ),
    ]
    .into_iter()
    .map(|(name, status)| StageOutcome { name, status })
    .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_check_skip_contract_diagnostic")?;

    let default = check(&CheckFlags::default());
    println!("default:        contract stage = {:?}", default[9].status);

    let skip_contract = check(&CheckFlags {
        skip_contract: true,
        ..Default::default()
    });
    println!(
        "--skip-contract: contract stage = {:?}",
        skip_contract[9].status
    );

    let skip_both = check(&CheckFlags {
        skip_contract: true,
        skip_signature: true,
    });
    println!(
        "--skip-contract --skip-signature: stages 9,10 = {:?}, {:?}",
        skip_both[8].status, skip_both[9].status
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_runs_contract_stage() {
        let report = check(&CheckFlags::default());
        assert_eq!(report[9].name, "contract");
        assert_eq!(report[9].status, StageStatus::Pass);
    }

    #[test]
    fn skip_contract_marks_stage_as_skipped_not_pass() {
        // Critical: a skipped stage must NOT report Pass — that would let
        // a model with a broken contract masquerade as healthy.
        let report = check(&CheckFlags {
            skip_contract: true,
            ..Default::default()
        });
        assert!(matches!(report[9].status, StageStatus::Skipped(_)));
        assert_ne!(report[9].status, StageStatus::Pass);
    }

    #[test]
    fn skip_contract_preserves_other_nine_stages() {
        let report = check(&CheckFlags {
            skip_contract: true,
            ..Default::default()
        });
        for stage in &report[..9] {
            assert_eq!(
                stage.status,
                StageStatus::Pass,
                "stage {} regressed",
                stage.name
            );
        }
    }

    #[test]
    fn skip_flags_are_independent() {
        let report = check(&CheckFlags {
            skip_contract: true,
            skip_signature: true,
        });
        assert!(matches!(report[8].status, StageStatus::Skipped(_)));
        assert!(matches!(report[9].status, StageStatus::Skipped(_)));
    }
}
