//! # Contracts-Macros YAML String Truncate Audit
//!
//! Detect YAML string values exceeding configured max length.
//! Returns sorted offending key names and the longest length seen.
//!
//! Demonstrates the **CMM.154** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint `line-length`; OWASP A03 input-validation
//!  string bound enforcement.
//!
//! Run with: cargo run --example contracts_macros_yaml_string_truncate
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TruncateVerdict {
    Ok {
        offending_keys: Vec<String>,
        max_length_seen: u32,
    },
    InvalidConfig,
}

pub fn audit(values: &[(&str, &str)], max_len: u32) -> TruncateVerdict {
    if values.is_empty() || max_len == 0 {
        return TruncateVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = values
        .iter()
        .filter(|(_, v)| v.chars().count() as u32 > max_len)
        .map(|(k, _)| (*k).to_string())
        .collect();
    offenders.sort();
    let longest = values
        .iter()
        .map(|(_, v)| v.chars().count() as u32)
        .max()
        .unwrap_or(0);
    TruncateVerdict::Ok {
        offending_keys: offenders,
        max_length_seen: longest,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_string_truncate")?;

    let values = [("name", "alice"), ("bio", "very long bio over the limit")];
    println!("max-10: {:?}", audit(&values, 10));
    println!("invalid: {:?}", audit(&[], 10));
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
    fn within_limit_no_offender() {
        let v = audit(&[("k", "abc")], 10);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn over_limit_flagged() {
        let v = audit(&[("k", "0123456789x")], 10);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["k".to_string()]);
        }
    }

    #[test]
    fn at_limit_in_band() {
        let v = audit(&[("k", "0123456789")], 10);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 10), TruncateVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(audit(&[("k", "v")], 0), TruncateVerdict::InvalidConfig);
    }

    #[test]
    fn max_length_seen_correct() {
        let v = audit(&[("a", "abc"), ("b", "abcdef")], 10);
        if let TruncateVerdict::Ok {
            max_length_seen, ..
        } = v
        {
            assert_eq!(max_length_seen, 6);
        }
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&[("zeta", "longvalueover"), ("alpha", "longvalueover")], 5);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(
                offending_keys,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("k", "v")], 10);
        let r2 = audit(&[("k", "v")], 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_chars_counted_correctly() {
        // "café" = 4 chars (not bytes)
        let v = audit(&[("k", "café")], 4);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn empty_value_handled() {
        let v = audit(&[("k", "")], 10);
        if let TruncateVerdict::Ok {
            max_length_seen, ..
        } = v
        {
            assert_eq!(max_length_seen, 0);
        }
    }

    #[test]
    fn many_values_handled() {
        let values: Vec<(&str, &str)> = (0..30).map(|_| ("k", "longvalueover")).collect();
        let v = audit(&values, 5);
        if let TruncateVerdict::Ok { offending_keys, .. } = v {
            // No dedup; each over-limit value is one offender entry.
            assert_eq!(offending_keys.len(), 30);
        }
    }
}
