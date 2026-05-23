//! # Contracts-Macros Recipe ID Format
//!
//! Validate recipe IDs match the canonical pattern `<PREFIX>.<NUM>`
//! where PREFIX is 2-3 uppercase letters and NUM is one or more
//! digits. Returns each ID's verdict (Ok, BadPrefix, BadNumber).
//!
//! Demonstrates the **CMM.74** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 1738 §2.1 BNF; semver naming conventions.
//!
//! Run with: cargo run --example contracts_macros_recipe_id_format
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum IdStatus {
    Ok,
    BadPrefix,
    BadNumber,
    NoSeparator,
}

#[derive(Debug, PartialEq)]
pub enum FormatVerdict {
    Ok { per_id: Vec<(String, IdStatus)> },
    InvalidConfig,
}

pub fn validate(ids: &[&str]) -> FormatVerdict {
    if ids.is_empty() {
        return FormatVerdict::InvalidConfig;
    }
    let mut per_id: Vec<(String, IdStatus)> = Vec::with_capacity(ids.len());
    for id in ids {
        let status = classify(id);
        per_id.push(((*id).to_string(), status));
    }
    FormatVerdict::Ok { per_id }
}

fn classify(id: &str) -> IdStatus {
    let parts: Vec<&str> = id.split('.').collect();
    if parts.len() != 2 {
        return IdStatus::NoSeparator;
    }
    let prefix = parts[0];
    let number = parts[1];
    if !(2..=3).contains(&prefix.chars().count()) || !prefix.chars().all(|c| c.is_ascii_uppercase())
    {
        return IdStatus::BadPrefix;
    }
    if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {
        return IdStatus::BadNumber;
    }
    IdStatus::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_id_format")?;

    let ids = ["TUI.55", "MC.123", "bad", "tui.5", "TUI.x"];
    println!("audit: {:?}", validate(&ids));
    println!("invalid: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn well_formed_id_ok() {
        let v = validate(&["TUI.55"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::Ok);
        }
    }

    #[test]
    fn three_letter_prefix_ok() {
        let v = validate(&["CMM.123"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::Ok);
        }
    }

    #[test]
    fn lowercase_prefix_bad() {
        let v = validate(&["tui.5"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::BadPrefix);
        }
    }

    #[test]
    fn one_letter_prefix_bad() {
        let v = validate(&["A.5"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::BadPrefix);
        }
    }

    #[test]
    fn four_letter_prefix_bad() {
        let v = validate(&["ABCD.5"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::BadPrefix);
        }
    }

    #[test]
    fn non_digit_number_bad() {
        let v = validate(&["TUI.x"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::BadNumber);
        }
    }

    #[test]
    fn empty_number_bad() {
        let v = validate(&["TUI."]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::BadNumber);
        }
    }

    #[test]
    fn no_dot_no_separator() {
        let v = validate(&["TUI55"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id[0].1, IdStatus::NoSeparator);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(validate(&[]), FormatVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&["TUI.55"]);
        let r2 = validate(&["TUI.55"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_preserved() {
        let v = validate(&["TUI.55", "MC.1", "bad"]);
        if let FormatVerdict::Ok { per_id } = v {
            assert_eq!(per_id.len(), 3);
        }
    }
}
