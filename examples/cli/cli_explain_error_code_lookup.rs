//! # apr explain — Error Code Lookup
//!
//! `apr explain <CODE>` (e.g., `apr explain GH-186`) returns a long-form
//! explanation, root cause, and remediation pointer for known error
//! codes. This recipe builds the codepath catalog and asserts the
//! contract: every code has all four fields, lookup is case-insensitive,
//! and unknown codes return None (not a silent default).
//!
//! Demonstrates the **EXPLAIN.7** recipe for PMAT-099 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender error-code spec
//!
//! Run with: cargo run --example cli_explain_error_code_lookup
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorEntry {
    pub code: &'static str,
    pub summary: &'static str,
    pub root_cause: &'static str,
    pub remediation: &'static str,
}

const CATALOG: &[ErrorEntry] = &[
    ErrorEntry {
        code: "GH-186",
        summary: "PAD token flood — model emits the same id every step",
        root_cause: "lm_head weight matrix transposed (in_dim ↔ out_dim swap)",
        remediation: "use `apr rosetta diff-tensors` to detect; re-export with correct layout",
    },
    ErrorEntry {
        code: "GH-188",
        summary: "Layout mismatch detected during conversion",
        root_cause: "GGML stores [in, out]; framework expects [out, in]",
        remediation: "specify --layout transpose during conversion",
    },
    ErrorEntry {
        code: "GH-223",
        summary: "Import without config.json produces garbage",
        root_cause: "rope_theta and other hyperparams cannot be inferred from shapes",
        remediation: "supply config.json or pass --allow-no-config and accept the risk",
    },
    ErrorEntry {
        code: "PMAT-237",
        summary: "Tensor contract validation slow on large models",
        root_cause: "per-tensor schema check is O(n_tensors × n_layers)",
        remediation: "use --skip-contract for diagnostic-only runs",
    },
    ErrorEntry {
        code: "F-PERF-042",
        summary: "GPU speedup below threshold",
        root_cause: "CPU runtime exceeded GPU runtime / target_speedup",
        remediation: "check kernel dispatch via apr ptx-map; verify FA path used",
    },
];

pub fn explain(code: &str) -> Option<&'static ErrorEntry> {
    let upper = code.to_ascii_uppercase();
    CATALOG.iter().find(|e| e.code.eq_ignore_ascii_case(&upper))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_error_code_lookup")?;

    for code in ["GH-186", "gh-188", "PMAT-237", "F-PERF-042", "GH-9999"] {
        match explain(code) {
            Some(e) => {
                println!("=== {code} ===");
                println!("  {}", e.summary);
                println!("  Root cause:  {}", e.root_cause);
                println!("  Remediation: {}", e.remediation);
            }
            None => println!("=== {code}  →  unknown error code"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_code_returns_entry() {
        let e = explain("GH-186").unwrap();
        assert_eq!(e.code, "GH-186");
        assert!(!e.summary.is_empty());
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let lower = explain("gh-186").unwrap();
        let upper = explain("GH-186").unwrap();
        assert_eq!(lower.code, upper.code);
    }

    #[test]
    fn unknown_code_returns_none() {
        // Critical: don't pick a default — operator's typo could mask a real
        // error category they want to look up.
        assert!(explain("GH-9999").is_none());
        assert!(explain("").is_none());
    }

    #[test]
    fn every_entry_has_all_four_fields() {
        for e in CATALOG {
            assert!(!e.code.is_empty());
            assert!(!e.summary.is_empty());
            assert!(!e.root_cause.is_empty());
            assert!(!e.remediation.is_empty());
        }
    }

    #[test]
    fn catalog_codes_are_unique() {
        // No duplicate entries — would make lookup ambiguous.
        let codes: std::collections::HashSet<&str> = CATALOG.iter().map(|e| e.code).collect();
        assert_eq!(codes.len(), CATALOG.len());
    }

    #[test]
    fn catalog_includes_critical_codes() {
        // Sanity: the most-cited bugs in MEMORY.md must be in the catalog.
        for code in ["GH-186", "GH-188", "GH-223", "PMAT-237"] {
            assert!(explain(code).is_some(), "missing critical code: {code}");
        }
    }
}
