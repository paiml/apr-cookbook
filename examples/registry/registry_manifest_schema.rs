//! # Registry Manifest Schema Validator
//!
//! Registry manifests declare: name (kebab-case), version (semver),
//! aliases (string list), parents (URI list). Schema enforcement: name
//! must match regex; version must parse as semver; aliases unique;
//! parents resolvable URIs. This recipe builds the validator.
//!
//! Demonstrates the **REG.8** recipe for PMAT-129 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender REG-001 + Semver 2.0.0 spec.
//!
//! Run with: cargo run --example registry_manifest_schema
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum SchemaVerdict {
    Ok,
    InvalidName { reason: &'static str },
    InvalidVersion { value: String },
    DuplicateAlias { alias: String },
    InvalidParentUri { uri: String },
    MissingRequiredField { field: &'static str },
}

pub fn validate(name: &str, version: &str, aliases: &[&str], parents: &[&str]) -> SchemaVerdict {
    if name.is_empty() {
        return SchemaVerdict::MissingRequiredField { field: "name" };
    }
    if !is_kebab_case(name) {
        return SchemaVerdict::InvalidName {
            reason: "must be kebab-case (a-z0-9, separated by '-')",
        };
    }
    if version.is_empty() {
        return SchemaVerdict::MissingRequiredField { field: "version" };
    }
    if !is_semver(version) {
        return SchemaVerdict::InvalidVersion {
            value: version.into(),
        };
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for a in aliases {
        if !seen.insert(a) {
            return SchemaVerdict::DuplicateAlias {
                alias: (*a).to_string(),
            };
        }
    }
    for p in parents {
        if !is_uri(p) {
            return SchemaVerdict::InvalidParentUri {
                uri: (*p).to_string(),
            };
        }
    }
    SchemaVerdict::Ok
}

fn is_kebab_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.starts_with('-') || s.ends_with('-') {
        return false;
    }
    if s.contains("--") {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn is_semver(s: &str) -> bool {
    let core = s.split_once('-').map_or(s, |(c, _)| c);
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == 3 && parts.iter().all(|p| p.parse::<u32>().is_ok())
}

fn is_uri(s: &str) -> bool {
    s.contains("://")
        && s.split_once("://")
            .is_some_and(|(scheme, rest)| !scheme.is_empty() && !rest.is_empty())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_manifest_schema")?;

    let cases = [
        (
            "llama-3-8b",
            "1.0.0",
            &["latest"][..],
            &["hf://meta/llama"][..],
        ),
        ("BadName", "1.0.0", &[][..], &[][..]),
        ("ok", "v1.2.3", &[][..], &[][..]),
        ("ok", "1.0.0", &["a", "a"][..], &[][..]),
        ("ok", "1.0.0", &[][..], &["no-scheme"][..]),
        ("", "1.0.0", &[][..], &[][..]),
    ];
    for (n, v, a, p) in cases {
        println!(
            "{n}/{v} aliases={a:?} parents={p:?}  →  {:?}",
            validate(n, v, a, p)
        );
    }
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
    fn typical_manifest_passes() {
        assert_eq!(
            validate("llama-3-8b", "1.0.0", &["latest"], &["hf://meta/llama"]),
            SchemaVerdict::Ok
        );
    }

    #[test]
    fn empty_name_rejected() {
        assert_eq!(
            validate("", "1.0.0", &[], &[]),
            SchemaVerdict::MissingRequiredField { field: "name" }
        );
    }

    #[test]
    fn capital_name_rejected() {
        let v = validate("BadName", "1.0.0", &[], &[]);
        assert!(matches!(v, SchemaVerdict::InvalidName { .. }));
    }

    #[test]
    fn double_dash_rejected() {
        let v = validate("llama--bad", "1.0.0", &[], &[]);
        assert!(matches!(v, SchemaVerdict::InvalidName { .. }));
    }

    #[test]
    fn leading_dash_rejected() {
        let v = validate("-bad", "1.0.0", &[], &[]);
        assert!(matches!(v, SchemaVerdict::InvalidName { .. }));
    }

    #[test]
    fn non_semver_rejected() {
        let v = validate("ok", "v1.2.3", &[], &[]);
        assert!(matches!(v, SchemaVerdict::InvalidVersion { .. }));
        let v2 = validate("ok", "1.2", &[], &[]);
        assert!(matches!(v2, SchemaVerdict::InvalidVersion { .. }));
    }

    #[test]
    fn semver_with_prerelease_passes() {
        assert_eq!(validate("ok", "1.2.3-rc1", &[], &[]), SchemaVerdict::Ok);
    }

    #[test]
    fn duplicate_alias_rejected() {
        let v = validate("ok", "1.0.0", &["a", "b", "a"], &[]);
        assert!(matches!(v, SchemaVerdict::DuplicateAlias { .. }));
    }

    #[test]
    fn invalid_parent_uri_rejected() {
        let v = validate("ok", "1.0.0", &[], &["no-scheme"]);
        assert!(matches!(v, SchemaVerdict::InvalidParentUri { .. }));
    }

    #[test]
    fn empty_aliases_and_parents_pass() {
        assert_eq!(validate("ok", "1.0.0", &[], &[]), SchemaVerdict::Ok);
    }
}
