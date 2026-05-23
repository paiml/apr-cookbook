//! # Contracts-Macros Parametric Obligation Expander
//!
//! Expand a parametric obligation template (e.g.,
//! `accuracy_per_class::{class_id}`) over a set of parameter values.
//! Returns the expanded obligation list.
//!
//! Demonstrates the **CMM.51** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: parametric type expansion (Lean type-class instances).
//!
//! Run with: cargo run --example contracts_macros_obligation_parametric
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ExpandVerdict {
    Ok { expanded: Vec<String>, count: u32 },
    EmptyTemplate,
    NoPlaceholder,
    EmptyParameters,
}

pub fn expand(template: &str, placeholder: &str, values: &[&str]) -> ExpandVerdict {
    if template.is_empty() {
        return ExpandVerdict::EmptyTemplate;
    }
    if values.is_empty() {
        return ExpandVerdict::EmptyParameters;
    }
    if !template.contains(placeholder) {
        return ExpandVerdict::NoPlaceholder;
    }
    let expanded: Vec<String> = values
        .iter()
        .map(|v| template.replace(placeholder, v))
        .collect();
    let count = expanded.len() as u32;
    ExpandVerdict::Ok { expanded, count }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_parametric")?;

    println!(
        "expanded: {:?}",
        expand(
            "accuracy_per_class::{class_id}",
            "{class_id}",
            &["cat", "dog"]
        )
    );
    println!(
        "no placeholder: {:?}",
        expand("plain_obligation", "{x}", &["a"])
    );
    println!("empty template: {:?}", expand("", "{x}", &["a"]));
    println!("empty params: {:?}", expand("foo::{x}", "{x}", &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expander_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_expansion() {
        let v = expand("accuracy::{class}", "{class}", &["a", "b", "c"]);
        if let ExpandVerdict::Ok { expanded, count } = v {
            assert_eq!(count, 3);
            assert_eq!(
                expanded,
                vec![
                    "accuracy::a".to_string(),
                    "accuracy::b".to_string(),
                    "accuracy::c".to_string(),
                ]
            );
        }
    }

    #[test]
    fn empty_template_rejected() {
        assert_eq!(expand("", "{x}", &["a"]), ExpandVerdict::EmptyTemplate);
    }

    #[test]
    fn no_placeholder_in_template() {
        let v = expand("plain", "{x}", &["a"]);
        assert_eq!(v, ExpandVerdict::NoPlaceholder);
    }

    #[test]
    fn empty_params_rejected() {
        assert_eq!(
            expand("foo::{x}", "{x}", &[]),
            ExpandVerdict::EmptyParameters
        );
    }

    #[test]
    fn unicode_values_supported() {
        let v = expand("hello::{name}", "{name}", &["café", "résumé"]);
        if let ExpandVerdict::Ok { expanded, .. } = v {
            assert!(expanded.iter().any(|s| s == "hello::café"));
        }
    }

    #[test]
    fn multiple_placeholder_occurrences() {
        let v = expand("{x}/{x}", "{x}", &["a"]);
        if let ExpandVerdict::Ok { expanded, .. } = v {
            assert_eq!(expanded[0], "a/a");
        }
    }

    #[test]
    fn single_value() {
        let v = expand("eq_{n}", "{n}", &["1"]);
        if let ExpandVerdict::Ok { expanded, .. } = v {
            assert_eq!(expanded, vec!["eq_1".to_string()]);
        }
    }

    #[test]
    fn many_values() {
        let values: Vec<&str> = (0..50).map(|_| "x").collect();
        let v = expand("o_{x}", "{x}", &values);
        if let ExpandVerdict::Ok { count, .. } = v {
            assert_eq!(count, 50);
        }
    }

    #[test]
    fn placeholder_with_unusual_chars() {
        let v = expand("foo::%V%", "%V%", &["bar"]);
        if let ExpandVerdict::Ok { expanded, .. } = v {
            assert_eq!(expanded[0], "foo::bar");
        }
    }

    #[test]
    fn deterministic() {
        let a = expand("e::{x}", "{x}", &["a", "b"]);
        let b = expand("e::{x}", "{x}", &["a", "b"]);
        assert_eq!(a, b);
    }
}
