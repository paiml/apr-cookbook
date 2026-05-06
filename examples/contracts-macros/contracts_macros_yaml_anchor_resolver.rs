//! # Contracts-Macros YAML Anchor Resolver
//!
//! Resolve `&anchor` definitions and `*alias` references in a list of
//! YAML name/value pairs. Returns the resolved-value list, or an
//! error if an alias points to an unknown anchor.
//!
//! Demonstrates the **CMM.46** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 anchor/alias spec (3.2.2.2).
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_resolver
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum AnchorVerdict {
    Resolved { entries: Vec<(String, String)> },
    UnknownAlias { alias: String },
    DuplicateAnchor { name: String },
    EmptyInput,
}

pub fn resolve(input: &[(&str, &str)]) -> AnchorVerdict {
    if input.is_empty() {
        return AnchorVerdict::EmptyInput;
    }
    let mut anchors: BTreeMap<&str, String> = BTreeMap::new();
    let mut output: Vec<(String, String)> = Vec::with_capacity(input.len());
    for (key, raw_value) in input {
        let value = raw_value.trim();
        if let Some(rest) = value.strip_prefix('&') {
            let (anchor_name, payload) = rest.split_once(' ').unwrap_or((rest, ""));
            if anchors.contains_key(anchor_name) {
                return AnchorVerdict::DuplicateAnchor {
                    name: anchor_name.to_string(),
                };
            }
            anchors.insert(anchor_name, payload.to_string());
            output.push(((*key).to_string(), payload.to_string()));
        } else if let Some(alias_name) = value.strip_prefix('*') {
            let Some(payload) = anchors.get(alias_name) else {
                return AnchorVerdict::UnknownAlias {
                    alias: alias_name.to_string(),
                };
            };
            output.push(((*key).to_string(), payload.clone()));
        } else {
            output.push(((*key).to_string(), value.to_string()));
        }
    }
    AnchorVerdict::Resolved { entries: output }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_resolver")?;

    let typical = [
        ("base", "&base hello"),
        ("ref", "*base"),
        ("plain", "world"),
    ];
    println!("typical: {:?}", resolve(&typical));

    let unknown = [("ref", "*missing")];
    println!("unknown: {:?}", resolve(&unknown));

    let dup = [("a", "&x foo"), ("b", "&x bar")];
    println!("duplicate: {:?}", resolve(&dup));

    println!("empty: {:?}", resolve(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn anchor_then_alias_resolves() {
        let v = resolve(&[("a", "&base hello"), ("b", "*base")]);
        if let AnchorVerdict::Resolved { entries } = v {
            assert_eq!(entries[0].1, "hello");
            assert_eq!(entries[1].1, "hello");
        }
    }

    #[test]
    fn plain_value_passthrough() {
        let v = resolve(&[("k", "plain")]);
        if let AnchorVerdict::Resolved { entries } = v {
            assert_eq!(entries[0].1, "plain");
        }
    }

    #[test]
    fn unknown_alias_rejected() {
        let v = resolve(&[("ref", "*ghost")]);
        if let AnchorVerdict::UnknownAlias { alias } = v {
            assert_eq!(alias, "ghost");
        }
    }

    #[test]
    fn duplicate_anchor_rejected() {
        let v = resolve(&[("a", "&x foo"), ("b", "&x bar")]);
        if let AnchorVerdict::DuplicateAnchor { name } = v {
            assert_eq!(name, "x");
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(resolve(&[]), AnchorVerdict::EmptyInput);
    }

    #[test]
    fn anchor_without_payload() {
        let v = resolve(&[("a", "&empty")]);
        if let AnchorVerdict::Resolved { entries } = v {
            assert_eq!(entries[0].1, "");
        }
    }

    #[test]
    fn order_preserved() {
        let v = resolve(&[("z", "1"), ("a", "2")]);
        if let AnchorVerdict::Resolved { entries } = v {
            assert_eq!(entries[0].0, "z");
            assert_eq!(entries[1].0, "a");
        }
    }

    #[test]
    fn forward_alias_unknown() {
        // Alias before anchor → unknown.
        let v = resolve(&[("b", "*x"), ("a", "&x foo")]);
        assert!(matches!(v, AnchorVerdict::UnknownAlias { .. }));
    }

    #[test]
    fn whitespace_value_trimmed() {
        let v = resolve(&[("k", "  trimmed  ")]);
        if let AnchorVerdict::Resolved { entries } = v {
            assert_eq!(entries[0].1, "trimmed");
        }
    }

    #[test]
    fn deterministic() {
        let input = [("a", "&x foo"), ("b", "*x")];
        let a = resolve(&input);
        let b = resolve(&input);
        assert_eq!(a, b);
    }
}
