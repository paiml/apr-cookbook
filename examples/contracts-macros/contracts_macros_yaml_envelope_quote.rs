//! # Contracts-Macros YAML Envelope Quote Audit
//!
//! Verify that envelope-style YAML values containing special tokens
//! are properly quoted (e.g., values with colons, hashes, leading
//! dashes). Returns sorted offending keys.
//!
//! Demonstrates the **CMM.175** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6.4.1 plain-scalar safety; ansible playbook
//!  quoting conventions.
//!
//! Run with: cargo run --example contracts_macros_yaml_envelope_quote
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EnvelopeQuoteVerdict {
    Ok {
        offending_keys: Vec<String>,
        clean: bool,
    },
    InvalidConfig,
}

/// Items: (key, value, is_quoted_in_source).
pub fn audit(items: &[(&str, &str, bool)]) -> EnvelopeQuoteVerdict {
    if items.is_empty() {
        return EnvelopeQuoteVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    for (key, value, quoted) in items {
        if needs_quoting(value) && !quoted {
            offenders.push((*key).to_string());
        }
    }
    offenders.sort();
    offenders.dedup();
    let clean = offenders.is_empty();
    EnvelopeQuoteVerdict::Ok {
        offending_keys: offenders,
        clean,
    }
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.contains(": ")
        || s.contains(" #")
        || s.starts_with('-')
        || s.starts_with('#')
        || s.starts_with('?')
        || s.starts_with('!')
        || s.starts_with('&')
        || s.starts_with('*')
        || s.starts_with('|')
        || s.starts_with('>')
        || s.starts_with('@')
        || s.contains('\n')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_envelope_quote")?;

    let items = [
        ("k1", "simple", false),
        ("k2", "has: colon", false),
        ("k3", "\"safe-quoted\"", true),
    ];
    println!("audit: {:?}", audit(&items));
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
    fn safe_value_no_offender() {
        let v = audit(&[("k", "simple", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn unsafe_unquoted_flagged() {
        let v = audit(&[("k", "has: colon", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn unsafe_quoted_ok() {
        let v = audit(&[("k", "has: colon", true)]);
        if let EnvelopeQuoteVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), EnvelopeQuoteVerdict::InvalidConfig);
    }

    #[test]
    fn leading_dash_flagged() {
        let v = audit(&[("k", "-1", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn leading_at_flagged() {
        let v = audit(&[("k", "@reserved", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("k", "v", false)]);
        let r2 = audit(&[("k", "v", false)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted_dedup() {
        let v = audit(&[("zeta", "?", false), ("alpha", "?", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(
                offending_keys,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn empty_value_safe() {
        let v = audit(&[("k", "", false)]);
        if let EnvelopeQuoteVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }

    #[test]
    fn newline_in_value_flagged() {
        let v = audit(&[("k", "line1\nline2", false)]);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str, bool)> = (0..30).map(|_| ("k", "?", false)).collect();
        let v = audit(&items);
        if let EnvelopeQuoteVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys.len(), 1);
        }
    }

    #[test]
    fn unicode_value_no_false_positive() {
        let v = audit(&[("k", "café", false)]);
        if let EnvelopeQuoteVerdict::Ok { clean, .. } = v {
            assert!(clean);
        }
    }
}
