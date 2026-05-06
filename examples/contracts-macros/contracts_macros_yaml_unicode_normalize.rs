//! # Contracts-Macros YAML Unicode Normalize
//!
//! Audit YAML keys for Unicode normalization: flag any key whose
//! bytes contain non-ASCII characters (heuristic for NFD/mixed),
//! since contract YAMLs should use ASCII-only keys for portability
//! and deterministic hashing.
//!
//! Demonstrates the **CMM.65** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Unicode TR15 (NFC vs NFD); YAML 1.2 spec §5.1.
//!
//! Run with: cargo run --example contracts_macros_yaml_unicode_normalize
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub struct OffendingKey {
    pub key: String,
    pub byte_count: usize,
    pub char_count: usize,
}

#[derive(Debug, PartialEq)]
pub enum NormalizeVerdict {
    Ok {
        ascii_only: bool,
        offending: Vec<OffendingKey>,
    },
    InvalidConfig,
}

pub fn audit(keys: &[&str]) -> NormalizeVerdict {
    if keys.is_empty() {
        return NormalizeVerdict::InvalidConfig;
    }
    let mut offending: Vec<OffendingKey> = Vec::new();
    for key in keys {
        if !key.is_ascii() {
            offending.push(OffendingKey {
                key: (*key).to_string(),
                byte_count: key.len(),
                char_count: key.chars().count(),
            });
        }
    }
    NormalizeVerdict::Ok {
        ascii_only: offending.is_empty(),
        offending,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_unicode_normalize")?;

    let clean = ["preconditions", "postconditions", "lean_theorem"];
    println!("clean: {:?}", audit(&clean));
    let mixed = ["preconditions", "café_count", "résumé"];
    println!("mixed: {:?}", audit(&mixed));
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
    fn pure_ascii_passes() {
        let keys = ["alpha", "beta", "gamma"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(ascii_only);
        }
    }

    #[test]
    fn unicode_key_flagged() {
        let keys = ["café"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(!ascii_only);
        }
    }

    #[test]
    fn unicode_key_byte_vs_char_diff() {
        let keys = ["café"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { offending, .. } = v {
            // 'é' is 2 bytes in UTF-8 → byte_count > char_count.
            assert!(offending[0].byte_count > offending[0].char_count);
        }
    }

    #[test]
    fn mixed_only_unicode_flagged() {
        let keys = ["alpha", "résumé", "beta"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { offending, .. } = v {
            assert_eq!(offending.len(), 1);
            assert_eq!(offending[0].key, "résumé");
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(audit(&[]), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn single_ascii_key_works() {
        let keys = ["foo"];
        let v = audit(&keys);
        assert_eq!(
            v,
            NormalizeVerdict::Ok {
                ascii_only: true,
                offending: vec![]
            }
        );
    }

    #[test]
    fn ascii_with_underscores_works() {
        let keys = ["snake_case_key", "kebab-case-key"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(ascii_only);
        }
    }

    #[test]
    fn ascii_digits_work() {
        let keys = ["v1", "v2_alpha"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(ascii_only);
        }
    }

    #[test]
    fn cjk_flagged() {
        let keys = ["日本語"];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(!ascii_only);
        }
    }

    #[test]
    fn deterministic() {
        let keys = ["alpha", "beta"];
        let a = audit(&keys);
        let b = audit(&keys);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_string_key_is_ascii() {
        let keys = [""];
        let v = audit(&keys);
        if let NormalizeVerdict::Ok { ascii_only, .. } = v {
            assert!(ascii_only);
        }
    }
}
