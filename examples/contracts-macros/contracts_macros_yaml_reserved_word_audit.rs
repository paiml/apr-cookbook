//! # Contracts-Macros YAML Reserved Word Audit
//!
//! Flag YAML keys that are also YAML scalar literals (true, false,
//! null, yes, no, on, off, ~) — these need quoting to be parsed
//! as strings.
//!
//! Demonstrates the **CMM.134** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.1 boolean ambiguity (Norway problem); YAML 1.2
//!  spec §10.3 scalar resolution.
//!
//! Run with: cargo run --example contracts_macros_yaml_reserved_word_audit
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReservedVerdict {
    Ok {
        offending: Vec<String>,
        ok_count: u32,
    },
    InvalidConfig,
}

const RESERVED: &[&str] = &[
    "true", "false", "null", "yes", "no", "on", "off", "~", "TRUE", "FALSE", "NULL", "YES", "NO",
    "ON", "OFF", "True", "False", "Null", "Yes", "No", "On", "Off", "y", "n", "Y", "N",
];

pub fn audit(keys: &[&str]) -> ReservedVerdict {
    if keys.is_empty() {
        return ReservedVerdict::InvalidConfig;
    }
    let mut offending: Vec<String> = Vec::new();
    let mut ok_count = 0u32;
    for key in keys {
        if RESERVED.contains(key) {
            offending.push((*key).to_string());
        } else {
            ok_count += 1;
        }
    }
    offending.sort();
    offending.dedup();
    ReservedVerdict::Ok {
        offending,
        ok_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_reserved_word_audit")?;

    let keys = ["name", "yes", "version", "no", "country"];
    println!("audit: {:?}", audit(&keys));
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
    fn safe_key_no_offender() {
        let v = audit(&["name", "version"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn yes_flagged() {
        let v = audit(&["yes"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["yes".to_string()]);
        }
    }

    #[test]
    fn norway_problem_no_flagged() {
        // Norway country code "no" is a famous YAML 1.1 trap.
        let v = audit(&["no"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["no".to_string()]);
        }
    }

    #[test]
    fn null_flagged() {
        let v = audit(&["null"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["null".to_string()]);
        }
    }

    #[test]
    fn on_off_flagged() {
        let v = audit(&["on", "off"]);
        if let ReservedVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 0);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), ReservedVerdict::InvalidConfig);
    }

    #[test]
    fn case_variants_flagged() {
        let v = audit(&["True", "FALSE"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending.len(), 2);
        }
    }

    #[test]
    fn ok_count_correct() {
        let v = audit(&["safe", "yes", "another"]);
        if let ReservedVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["yes"]);
        let r2 = audit(&["yes"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offending_sorted() {
        let v = audit(&["yes", "no"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["no".to_string(), "yes".to_string()]);
        }
    }

    #[test]
    fn duplicate_offender_dedup() {
        let v = audit(&["yes", "yes"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["yes".to_string()]);
        }
    }

    #[test]
    fn tilde_flagged() {
        let v = audit(&["~"]);
        if let ReservedVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["~".to_string()]);
        }
    }
}
