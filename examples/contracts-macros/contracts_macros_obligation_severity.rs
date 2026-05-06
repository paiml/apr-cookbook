//! # Contracts-Macros Obligation Severity Classifier
//!
//! Classify a contract obligation by impact:
//!   Blocking — production traffic must abort
//!   Required — release blocked, but staging traffic OK
//!   Advisory — log + monitor
//!   NotApplicable — n/a phase
//!
//! Demonstrates the **CMM.21** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLO error budgets + severity ladders (Google SRE workbook).
//!
//! Run with: cargo run --example contracts_macros_obligation_severity
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Production,
    Staging,
    Development,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Blocking,
    Required,
    Advisory,
    NotApplicable,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Pick {
        severity: Severity,
        reason: &'static str,
    },
}

pub fn classify(phase: Phase, fail_rate_pct: f64) -> ClassifyVerdict {
    let bounded = fail_rate_pct.clamp(0.0, 100.0);
    match phase {
        Phase::Disabled => ClassifyVerdict::Pick {
            severity: Severity::NotApplicable,
            reason: "phase is disabled",
        },
        Phase::Production if bounded >= 1.0 => ClassifyVerdict::Pick {
            severity: Severity::Blocking,
            reason: "production fail rate ≥ 1%",
        },
        Phase::Production => ClassifyVerdict::Pick {
            severity: Severity::Required,
            reason: "production fail rate below abort threshold",
        },
        Phase::Staging if bounded >= 5.0 => ClassifyVerdict::Pick {
            severity: Severity::Required,
            reason: "staging fail rate ≥ 5%",
        },
        Phase::Staging => ClassifyVerdict::Pick {
            severity: Severity::Advisory,
            reason: "staging fail rate within budget",
        },
        Phase::Development => ClassifyVerdict::Pick {
            severity: Severity::Advisory,
            reason: "development always advisory",
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_severity")?;

    println!("prod blocking: {:?}", classify(Phase::Production, 2.5));
    println!("prod required: {:?}", classify(Phase::Production, 0.5));
    println!("staging required: {:?}", classify(Phase::Staging, 6.0));
    println!("staging advisory: {:?}", classify(Phase::Staging, 1.0));
    println!("dev: {:?}", classify(Phase::Development, 50.0));
    println!("disabled: {:?}", classify(Phase::Disabled, 99.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn prod_high_blocking() {
        let v = classify(Phase::Production, 2.5);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Blocking);
        }
    }

    #[test]
    fn prod_low_required() {
        let v = classify(Phase::Production, 0.5);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Required);
        }
    }

    #[test]
    fn staging_high_required() {
        let v = classify(Phase::Staging, 6.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Required);
        }
    }

    #[test]
    fn staging_low_advisory() {
        let v = classify(Phase::Staging, 1.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Advisory);
        }
    }

    #[test]
    fn dev_always_advisory() {
        let v = classify(Phase::Development, 50.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Advisory);
        }
    }

    #[test]
    fn disabled_not_applicable() {
        let v = classify(Phase::Disabled, 99.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::NotApplicable);
        }
    }

    #[test]
    fn boundary_at_1_pct_blocking() {
        let v = classify(Phase::Production, 1.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Blocking);
        }
    }

    #[test]
    fn boundary_at_5_pct_required() {
        let v = classify(Phase::Staging, 5.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Required);
        }
    }

    #[test]
    fn negative_clamps_to_zero() {
        let v = classify(Phase::Production, -10.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Required);
        }
    }

    #[test]
    fn over_100_clamps() {
        let v = classify(Phase::Production, 200.0);
        if let ClassifyVerdict::Pick { severity, .. } = v {
            assert_eq!(severity, Severity::Blocking);
        }
    }

    #[test]
    fn deterministic() {
        let a = classify(Phase::Production, 2.5);
        let b = classify(Phase::Production, 2.5);
        assert_eq!(a, b);
    }
}
