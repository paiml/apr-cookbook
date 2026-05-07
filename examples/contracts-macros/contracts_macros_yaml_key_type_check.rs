//! # Contracts-Macros YAML Key Type Check
//!
//! Verify all YAML keys are strings (not numbers or booleans).
//! Returns sorted offending keys.
//!
//! Demonstrates the **CMM.202** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6.4.2 implicit-typing rules; libyaml strict-
//!  string-keys mode.
//!
//! Run with: cargo run --example contracts_macros_yaml_key_type_check
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KeyTypeVerdict {
    Ok {
        offending_keys: Vec<String>,
        string_count: u32,
    },
    InvalidConfig,
}

pub fn check(keys: &[&str]) -> KeyTypeVerdict {
    if keys.is_empty() {
        return KeyTypeVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = keys
        .iter()
        .filter(|k| !is_string_key(k))
        .map(|k| (*k).to_string())
        .collect();
    offenders.sort();
    let strings = keys.iter().filter(|k| is_string_key(k)).count() as u32;
    KeyTypeVerdict::Ok {
        offending_keys: offenders,
        string_count: strings,
    }
}

fn is_string_key(k: &str) -> bool {
    if k.is_empty() {
        return false;
    }
    // Reject if parses as number or boolean.
    if k.parse::<f64>().is_ok() {
        return false;
    }
    if matches!(
        k,
        "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "~"
    ) {
        return false;
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_key_type_check")?;

    println!("clean: {:?}", check(&["name", "version"]));
    println!("numeric: {:?}", check(&["1", "name"]));
    println!("bool: {:?}", check(&["true", "name"]));
    println!("invalid: {:?}", check(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn string_key_no_offender() {
        let v = check(&["name"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn numeric_key_offender() {
        let v = check(&["42"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["42".to_string()]);
        }
    }

    #[test]
    fn boolean_key_offender() {
        let v = check(&["true"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["true".to_string()]);
        }
    }

    #[test]
    fn empty_key_offender() {
        let v = check(&[""]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), KeyTypeVerdict::InvalidConfig);
    }

    #[test]
    fn yaml_truthy_words_offender() {
        for k in &["yes", "no", "on", "off", "null", "~"] {
            let v = check(&[k]);
            if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
                assert!(offending_keys.contains(&k.to_string()));
            }
        }
    }

    #[test]
    fn float_key_offender() {
        let v = check(&["3.14"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["3.14".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["name"]);
        let r2 = check(&["name"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&["zeta_safe", "true", "0", "alpha_safe"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            for w in offending_keys.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn string_count_correct() {
        let v = check(&["name", "true", "version"]);
        if let KeyTypeVerdict::Ok { string_count, .. } = v {
            assert_eq!(string_count, 2);
        }
    }

    #[test]
    fn many_keys_handled() {
        let keys: Vec<&str> = (0..30).map(|_| "key").collect();
        let v = check(&keys);
        if let KeyTypeVerdict::Ok { string_count, .. } = v {
            assert_eq!(string_count, 30);
        }
    }

    #[test]
    fn unicode_key_supported() {
        let v = check(&["café"]);
        if let KeyTypeVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }
}
