//! # apr qa — SafeTensors Parity Required When Format-Parity On
//!
//! `apr qa <FILE> --safetensors-path <P>` is REQUIRED when the
//! `format-parity` check is active (the default). Omitting it while the
//! check is on is a contract violation that the binary used to silently
//! skip — this recipe pins the validation rule so future regressions are
//! caught at the boundary.
//!
//! Demonstrates the **QA.5** recipe for PMAT-093 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender F-QUAL-032
//!
//! Run with: cargo run --example cli_qa_safetensors_parity_required
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone)]
pub struct QaInvocation {
    pub model: String,
    pub safetensors_path: Option<String>,
    pub skip_format_parity: bool,
}

#[derive(Debug, PartialEq)]
pub enum InvocationVerdict {
    Ok,
    MissingSafetensorsPath,
    SuperfluousSafetensorsPath, // path supplied but check skipped — usability warning
}

pub fn validate_invocation(inv: &QaInvocation) -> InvocationVerdict {
    match (inv.skip_format_parity, &inv.safetensors_path) {
        (false, None) => InvocationVerdict::MissingSafetensorsPath,
        (true, Some(_)) => InvocationVerdict::SuperfluousSafetensorsPath,
        _ => InvocationVerdict::Ok,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_safetensors_parity_required")?;

    let cases = [
        (
            "happy",
            QaInvocation {
                model: "model.apr".into(),
                safetensors_path: Some("model.safetensors".into()),
                skip_format_parity: false,
            },
        ),
        (
            "skip-parity-no-path",
            QaInvocation {
                model: "model.apr".into(),
                safetensors_path: None,
                skip_format_parity: true,
            },
        ),
        (
            "missing-path",
            QaInvocation {
                model: "model.apr".into(),
                safetensors_path: None,
                skip_format_parity: false,
            },
        ),
        (
            "superfluous-path",
            QaInvocation {
                model: "model.apr".into(),
                safetensors_path: Some("ignored.safetensors".into()),
                skip_format_parity: true,
            },
        ),
    ];

    println!("=== Recipe: cli_qa_safetensors_parity_required ===");
    for (label, inv) in cases {
        println!("{label:>22}  →  {:?}", validate_invocation(&inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_invocation_with_path_passes() {
        let inv = QaInvocation {
            model: "m.apr".into(),
            safetensors_path: Some("m.safetensors".into()),
            skip_format_parity: false,
        };
        assert_eq!(validate_invocation(&inv), InvocationVerdict::Ok);
    }

    #[test]
    fn missing_path_when_check_active_is_error() {
        let inv = QaInvocation {
            model: "m.apr".into(),
            safetensors_path: None,
            skip_format_parity: false,
        };
        assert_eq!(
            validate_invocation(&inv),
            InvocationVerdict::MissingSafetensorsPath
        );
    }

    #[test]
    fn skip_check_with_no_path_is_ok() {
        let inv = QaInvocation {
            model: "m.apr".into(),
            safetensors_path: None,
            skip_format_parity: true,
        };
        assert_eq!(validate_invocation(&inv), InvocationVerdict::Ok);
    }

    #[test]
    fn superfluous_path_with_skip_is_warned() {
        // Operator passed a path but also skipped the check that uses it.
        // Surface the inconsistency so the operator catches the typo.
        let inv = QaInvocation {
            model: "m.apr".into(),
            safetensors_path: Some("ignored.safetensors".into()),
            skip_format_parity: true,
        };
        assert_eq!(
            validate_invocation(&inv),
            InvocationVerdict::SuperfluousSafetensorsPath
        );
    }
}
