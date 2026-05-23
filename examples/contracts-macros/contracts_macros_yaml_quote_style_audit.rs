//! # Contracts-Macros YAML Quote Style Audit
//!
//! Audit YAML string values for inconsistent quote-style use:
//! mixing single (`'...'`) and double (`"..."`) within one file.
//!
//! Demonstrates the **CMM.101** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §7.3 (flow scalar styles); editorconfig
//!  consistency conventions.
//!
//! Run with: cargo run --example contracts_macros_yaml_quote_style_audit
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum QuoteStyle {
    Single,
    Double,
    None,
}

#[derive(Debug, PartialEq)]
pub enum QuoteVerdict {
    Ok {
        single_count: u32,
        double_count: u32,
        consistent: bool,
    },
    InvalidConfig,
}

pub fn audit(values: &[(&str, QuoteStyle)]) -> QuoteVerdict {
    if values.is_empty() {
        return QuoteVerdict::InvalidConfig;
    }
    let mut single = 0u32;
    let mut double = 0u32;
    for (_, style) in values {
        match style {
            QuoteStyle::Single => single += 1,
            QuoteStyle::Double => double += 1,
            QuoteStyle::None => {}
        }
    }
    let consistent = single == 0 || double == 0;
    QuoteVerdict::Ok {
        single_count: single,
        double_count: double,
        consistent,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_quote_style_audit")?;

    let values = [
        ("k1", QuoteStyle::Single),
        ("k2", QuoteStyle::Single),
        ("k3", QuoteStyle::Double),
    ];
    println!("mixed: {:?}", audit(&values));
    let consistent = [("k1", QuoteStyle::Single), ("k2", QuoteStyle::Single)];
    println!("consistent: {:?}", audit(&consistent));
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
    fn all_single_consistent() {
        let values = [("k1", QuoteStyle::Single), ("k2", QuoteStyle::Single)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn all_double_consistent() {
        let values = [("k1", QuoteStyle::Double), ("k2", QuoteStyle::Double)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn mixed_inconsistent() {
        let values = [("k1", QuoteStyle::Single), ("k2", QuoteStyle::Double)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(!consistent);
        }
    }

    #[test]
    fn none_kind_doesnt_break_consistency() {
        let values = [("k1", QuoteStyle::Single), ("k2", QuoteStyle::None)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), QuoteVerdict::InvalidConfig);
    }

    #[test]
    fn single_count_correct() {
        let values = [
            ("k1", QuoteStyle::Single),
            ("k2", QuoteStyle::Single),
            ("k3", QuoteStyle::Double),
        ];
        let v = audit(&values);
        if let QuoteVerdict::Ok { single_count, .. } = v {
            assert_eq!(single_count, 2);
        }
    }

    #[test]
    fn double_count_correct() {
        let values = [("k1", QuoteStyle::Double), ("k2", QuoteStyle::Double)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { double_count, .. } = v {
            assert_eq!(double_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let values = [("k1", QuoteStyle::Single)];
        let r1 = audit(&values);
        let r2 = audit(&values);
        assert_eq!(r1, r2);
    }

    #[test]
    fn all_none_consistent() {
        let values = [("k1", QuoteStyle::None), ("k2", QuoteStyle::None)];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(consistent);
        }
    }

    #[test]
    fn one_of_each_inconsistent() {
        let values = [
            ("k1", QuoteStyle::Single),
            ("k2", QuoteStyle::Double),
            ("k3", QuoteStyle::None),
        ];
        let v = audit(&values);
        if let QuoteVerdict::Ok { consistent, .. } = v {
            assert!(!consistent);
        }
    }

    #[test]
    fn sum_le_total() {
        let values = [
            ("k1", QuoteStyle::Single),
            ("k2", QuoteStyle::Double),
            ("k3", QuoteStyle::None),
        ];
        let v = audit(&values);
        if let QuoteVerdict::Ok {
            single_count,
            double_count,
            ..
        } = v
        {
            assert!(single_count + double_count <= 3);
        }
    }
}
