//! # Contracts-Macros YAML Quoted Keys Audit
//!
//! Detect YAML keys that need quoting (start with reserved chars or
//! contain special tokens). Returns sorted offending keys and a
//! count of normalizable keys.
//!
//! Demonstrates the **CMM.144** recipe for PMAT-205 (post-milestone).
//!
//! Citation: YAML 1.2 §6.4.1 plain-scalar production rules; libyaml
//!  scanner_check_plain start-char restrictions.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Run with: cargo run --example contracts_macros_yaml_quoted_keys_audit
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuotedKeysVerdict {
    Ok {
        offending_keys: Vec<String>,
        plain_safe_count: u32,
    },
    InvalidConfig,
}

pub fn audit(keys: &[&str]) -> QuotedKeysVerdict {
    if keys.is_empty() {
        return QuotedKeysVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut safe = 0u32;
    for k in keys {
        if needs_quoting(k) {
            offenders.push((*k).to_string());
        } else {
            safe += 1;
        }
    }
    offenders.sort();
    offenders.dedup();
    QuotedKeysVerdict::Ok {
        offending_keys: offenders,
        plain_safe_count: safe,
    }
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    let first = s.chars().next().unwrap();
    if "!&*-?:|>%@`,[]{}#\"'".contains(first) {
        return true;
    }
    s.contains(": ") || s.contains(" #") || s.starts_with(' ') || s.ends_with(' ')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_quoted_keys_audit")?;

    let keys = ["safe_key", ":bad_start", "has: colon", "trailing "];
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
        let v = audit(&["clean_key"]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn reserved_first_char_flagged() {
        let v = audit(&["?reserved"]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["?reserved".to_string()]);
        }
    }

    #[test]
    fn colon_space_flagged() {
        let v = audit(&["has: colon"]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["has: colon".to_string()]);
        }
    }

    #[test]
    fn trailing_space_flagged() {
        let v = audit(&["trail "]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["trail ".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), QuotedKeysVerdict::InvalidConfig);
    }

    #[test]
    fn empty_key_flagged() {
        let v = audit(&[""]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["".to_string()]);
        }
    }

    #[test]
    fn safe_count_correct() {
        let v = audit(&["safe1", ":bad", "safe2"]);
        if let QuotedKeysVerdict::Ok {
            plain_safe_count, ..
        } = v
        {
            assert_eq!(plain_safe_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a"]);
        let r2 = audit(&["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&["?zeta", "?alpha"]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(
                offending_keys,
                vec!["?alpha".to_string(), "?zeta".to_string()]
            );
        }
    }

    #[test]
    fn duplicates_dedup() {
        let v = audit(&["?a", "?a"]);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys.len(), 1);
        }
    }

    #[test]
    fn many_keys_handled() {
        let keys: Vec<&str> = (0..30).map(|_| "?bad").collect();
        let v = audit(&keys);
        if let QuotedKeysVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys.len(), 1);
        }
    }
}
