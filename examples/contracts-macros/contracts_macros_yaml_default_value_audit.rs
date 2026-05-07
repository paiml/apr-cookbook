//! # Contracts-Macros YAML Default Value Audit
//!
//! Flag suspicious default values in YAML config: empty strings,
//! null, "false" literal, or 0 for required-non-zero fields.
//! Returns offending key names.
//!
//! Demonstrates the **CMM.125** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: secure default conventions (NIST SP 800-160); JSON
//!  Schema `default` validation hints.
//!
//! Run with: cargo run --example contracts_macros_yaml_default_value_audit
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DefaultIssue {
    EmptyString,
    NullLiteral,
    FalseLiteral,
    ZeroValue,
}

#[derive(Debug, PartialEq)]
pub enum DefaultAuditVerdict {
    Ok {
        per_key: Vec<(String, Option<DefaultIssue>)>,
        suspicious_count: u32,
    },
    InvalidConfig,
}

pub fn audit(defaults: &[(&str, &str)]) -> DefaultAuditVerdict {
    if defaults.is_empty() {
        return DefaultAuditVerdict::InvalidConfig;
    }
    let mut per_key: Vec<(String, Option<DefaultIssue>)> = Vec::with_capacity(defaults.len());
    let mut suspicious_count = 0u32;
    for (key, value) in defaults {
        let issue = match *value {
            "" => Some(DefaultIssue::EmptyString),
            "null" | "~" => Some(DefaultIssue::NullLiteral),
            "false" => Some(DefaultIssue::FalseLiteral),
            "0" => Some(DefaultIssue::ZeroValue),
            _ => None,
        };
        if issue.is_some() {
            suspicious_count += 1;
        }
        per_key.push(((*key).to_string(), issue));
    }
    DefaultAuditVerdict::Ok {
        per_key,
        suspicious_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_default_value_audit")?;

    let defaults = [
        ("timeout", "30"),
        ("name", ""),
        ("retry", "0"),
        ("ssl", "false"),
    ];
    println!("audit: {:?}", audit(&defaults));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_string_flagged() {
        let v = audit(&[("name", "")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert_eq!(per_key[0].1, Some(DefaultIssue::EmptyString));
        }
    }

    #[test]
    fn null_flagged() {
        let v = audit(&[("name", "null")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert_eq!(per_key[0].1, Some(DefaultIssue::NullLiteral));
        }
    }

    #[test]
    fn tilde_null_flagged() {
        let v = audit(&[("name", "~")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert_eq!(per_key[0].1, Some(DefaultIssue::NullLiteral));
        }
    }

    #[test]
    fn false_flagged() {
        let v = audit(&[("name", "false")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert_eq!(per_key[0].1, Some(DefaultIssue::FalseLiteral));
        }
    }

    #[test]
    fn zero_flagged() {
        let v = audit(&[("name", "0")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert_eq!(per_key[0].1, Some(DefaultIssue::ZeroValue));
        }
    }

    #[test]
    fn good_value_no_issue() {
        let v = audit(&[("name", "production")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert!(per_key[0].1.is_none());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), DefaultAuditVerdict::InvalidConfig);
    }

    #[test]
    fn count_correct() {
        let v = audit(&[("a", ""), ("b", "ok"), ("c", "0")]);
        if let DefaultAuditVerdict::Ok {
            suspicious_count, ..
        } = v
        {
            assert_eq!(suspicious_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", "ok")]);
        let r2 = audit(&[("a", "ok")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn one_value_works() {
        let v = audit(&[("a", "production")]);
        if let DefaultAuditVerdict::Ok {
            suspicious_count, ..
        } = v
        {
            assert_eq!(suspicious_count, 0);
        }
    }

    #[test]
    fn space_only_not_empty() {
        let v = audit(&[("name", " ")]);
        if let DefaultAuditVerdict::Ok { per_key, .. } = v {
            assert!(per_key[0].1.is_none());
        }
    }
}
