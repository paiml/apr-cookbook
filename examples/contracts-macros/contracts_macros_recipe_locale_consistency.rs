//! # Contracts-Macros Recipe Locale Consistency
//!
//! Verify all locale files declare the same set of translation keys.
//! Returns sorted missing-key entries by locale.
//!
//! Demonstrates the **CMM.188** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: gettext .po file consistency; ICU MessageFormat key
//!  parity rules (Unicode TR35).
//!
//! Run with: cargo run --example contracts_macros_recipe_locale_consistency
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum LocaleVerdict {
    Ok {
        missing_per_locale: BTreeMap<String, Vec<String>>,
        complete_locales: u32,
    },
    InvalidConfig,
}

/// Items: (locale, keys present).
pub fn check(locales: &[(&str, Vec<&str>)]) -> LocaleVerdict {
    if locales.len() < 2 {
        return LocaleVerdict::InvalidConfig;
    }
    // Union of all keys = the canonical set.
    let mut canonical: BTreeSet<String> = BTreeSet::new();
    for (_, keys) in locales {
        for k in keys {
            canonical.insert((*k).to_string());
        }
    }
    let mut missing_per: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut complete = 0u32;
    for (locale, keys) in locales {
        let key_set: BTreeSet<String> = keys.iter().map(|k| (*k).to_string()).collect();
        let missing: Vec<String> = canonical.difference(&key_set).cloned().collect();
        if missing.is_empty() {
            complete += 1;
        } else {
            missing_per.insert((*locale).to_string(), missing);
        }
    }
    LocaleVerdict::Ok {
        missing_per_locale: missing_per,
        complete_locales: complete,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_locale_consistency")?;

    let locales = vec![
        ("en", vec!["greeting", "farewell"]),
        ("es", vec!["greeting"]),
    ];
    println!("check: {:?}", check(&locales));
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
    fn complete_locales_no_missing() {
        let locales = vec![("en", vec!["a", "b"]), ("es", vec!["a", "b"])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            assert!(missing_per_locale.is_empty());
        }
    }

    #[test]
    fn missing_key_flagged() {
        let locales = vec![("en", vec!["a", "b"]), ("es", vec!["a"])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            assert_eq!(missing_per_locale.get("es"), Some(&vec!["b".to_string()]));
        }
    }

    #[test]
    fn fewer_than_two_rejected() {
        let locales: Vec<(&str, Vec<&str>)> = vec![("en", vec!["a"])];
        assert_eq!(check(&locales), LocaleVerdict::InvalidConfig);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(check(&[]), LocaleVerdict::InvalidConfig);
    }

    #[test]
    fn complete_count_correct() {
        let locales = vec![
            ("en", vec!["a", "b"]),
            ("es", vec!["a", "b"]),
            ("fr", vec!["a"]),
        ];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            complete_locales, ..
        } = v
        {
            assert_eq!(complete_locales, 2);
        }
    }

    #[test]
    fn deterministic() {
        let locales = vec![("en", vec!["a"]), ("es", vec!["a"])];
        let r1 = check(&locales);
        let r2 = check(&locales);
        assert_eq!(r1, r2);
    }

    #[test]
    fn extra_keys_in_one_locale_others_missing() {
        let locales = vec![("en", vec!["a", "b", "c"]), ("es", vec!["a"])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            // "es" missing both "b" and "c"
            let es_missing = missing_per_locale.get("es").unwrap();
            assert_eq!(es_missing.len(), 2);
        }
    }

    #[test]
    fn unicode_key_supported() {
        let locales = vec![("en", vec!["café"]), ("fr", vec!["café"])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            assert!(missing_per_locale.is_empty());
        }
    }

    #[test]
    fn many_locales_handled() {
        let locales: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("loc", vec!["a"])).collect();
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            complete_locales, ..
        } = v
        {
            // Single canonical key "a"; all locales present → complete = 1 unique entry.
            assert!(complete_locales >= 1);
        }
    }

    #[test]
    fn empty_keys_locale_flagged() {
        let locales: Vec<(&str, Vec<&str>)> = vec![("en", vec!["a"]), ("es", vec![])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            assert!(missing_per_locale.contains_key("es"));
        }
    }

    #[test]
    fn case_sensitive_keys() {
        let locales = vec![("en", vec!["Hello"]), ("es", vec!["hello"])];
        let v = check(&locales);
        if let LocaleVerdict::Ok {
            missing_per_locale, ..
        } = v
        {
            // Both locales missing the other's key.
            assert_eq!(missing_per_locale.len(), 2);
        }
    }
}
