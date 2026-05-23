//! # apr validate --strict — Warning → Error Promoter
//!
//! `apr validate --strict` promotes warnings to errors. This recipe
//! builds the per-finding promoter and asserts the contract: existing
//! errors stay errors, warnings become errors under --strict, info
//! stays info either way.
//!
//! Demonstrates the **VALIDATE.12** recipe for PMAT-108 (apr validate coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender VALIDATE-002
//!
//! Run with: cargo run --example cli_validate_strict_warning_promoter
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub category: String,
    pub severity: Severity,
    pub message: String,
}

pub fn promote_under_strict(findings: &[Finding], strict: bool) -> Vec<Finding> {
    findings
        .iter()
        .map(|f| {
            let new_sev = match (f.severity, strict) {
                (Severity::Warning, true) => Severity::Error,
                (s, _) => s,
            };
            Finding {
                category: f.category.clone(),
                severity: new_sev,
                message: f.message.clone(),
            }
        })
        .collect()
}

pub fn count_errors(findings: &[Finding]) -> usize {
    findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .count()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_strict_warning_promoter")?;

    let findings = vec![
        Finding {
            category: "tokenizer".into(),
            severity: Severity::Info,
            message: "32k vocab".into(),
        },
        Finding {
            category: "provenance".into(),
            severity: Severity::Warning,
            message: "missing license".into(),
        },
        Finding {
            category: "integrity".into(),
            severity: Severity::Error,
            message: "CRC mismatch".into(),
        },
    ];

    println!("normal mode:");
    let normal = promote_under_strict(&findings, false);
    for f in &normal {
        println!("  {:?}: {}", f.severity, f.category);
    }
    println!("  errors: {}", count_errors(&normal));

    println!("\nstrict mode:");
    let strict = promote_under_strict(&findings, true);
    for f in &strict {
        println!("  {:?}: {}", f.severity, f.category);
    }
    println!("  errors: {}", count_errors(&strict));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_findings() -> Vec<Finding> {
        vec![
            Finding {
                category: "a".into(),
                severity: Severity::Info,
                message: "i".into(),
            },
            Finding {
                category: "b".into(),
                severity: Severity::Warning,
                message: "w".into(),
            },
            Finding {
                category: "c".into(),
                severity: Severity::Error,
                message: "e".into(),
            },
        ]
    }

    #[test]
    fn promoter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_mode_preserves_severity() {
        let f = sample_findings();
        let p = promote_under_strict(&f, false);
        assert_eq!(count_errors(&p), 1);
    }

    #[test]
    fn strict_mode_promotes_warnings_to_errors() {
        let f = sample_findings();
        let p = promote_under_strict(&f, true);
        assert_eq!(count_errors(&p), 2); // original error + promoted warning
    }

    #[test]
    fn strict_mode_does_not_promote_info() {
        let f = sample_findings();
        let p = promote_under_strict(&f, true);
        assert!(p.iter().any(|f| f.severity == Severity::Info));
    }

    #[test]
    fn empty_findings_yield_empty_output() {
        let p = promote_under_strict(&[], true);
        assert!(p.is_empty());
        assert_eq!(count_errors(&p), 0);
    }

    #[test]
    fn message_and_category_preserved_through_promotion() {
        let f = vec![Finding {
            category: "tokenizer".into(),
            severity: Severity::Warning,
            message: "vocab gap".into(),
        }];
        let p = promote_under_strict(&f, true);
        assert_eq!(p[0].category, "tokenizer");
        assert_eq!(p[0].message, "vocab gap");
        assert_eq!(p[0].severity, Severity::Error);
    }

    #[test]
    fn severity_ordering_natural() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }
}
