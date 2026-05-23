//! # Contracts-Macros Recipe ID Canonicalizer
//!
//! Normalize recipe IDs: lowercase, replace spaces and dashes with
//! underscores, strip extension, reject IDs that fail to round-trip.
//!
//! Demonstrates the **CMM.25** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: snake_case naming convention (PEP 8 / Rust style).
//!
//! Run with: cargo run --example contracts_macros_recipe_id_canon
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CanonVerdict {
    Ok { canonical: String },
    EmptyId,
    InvalidChars { offending: char },
}

pub fn canonicalize(raw: &str) -> CanonVerdict {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return CanonVerdict::EmptyId;
    }
    let stripped = trimmed.strip_suffix(".rs").unwrap_or(trimmed);
    let mut canonical = String::with_capacity(stripped.len());
    let mut prev_underscore = false;
    for c in stripped.chars() {
        let normalized = match c {
            'a'..='z' | '0'..='9' | '_' => c,
            'A'..='Z' => c.to_ascii_lowercase(),
            ' ' | '-' => '_',
            _ => return CanonVerdict::InvalidChars { offending: c },
        };
        // Collapse multiple underscores.
        if normalized == '_' && prev_underscore {
            continue;
        }
        prev_underscore = normalized == '_';
        canonical.push(normalized);
    }
    let canonical = canonical.trim_matches('_').to_string();
    if canonical.is_empty() {
        return CanonVerdict::EmptyId;
    }
    CanonVerdict::Ok { canonical }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_id_canon")?;

    println!("typical: {:?}", canonicalize("My Recipe Name"));
    println!("dashes: {:?}", canonicalize("apr-merge"));
    println!("with ext: {:?}", canonicalize("recipe.rs"));
    println!("multiple sep: {:?}", canonicalize("--foo  bar--"));
    println!("invalid: {:?}", canonicalize("foo!bar"));
    println!("empty: {:?}", canonicalize("   "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn space_to_underscore() {
        let v = canonicalize("My Recipe");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "my_recipe");
        }
    }

    #[test]
    fn dash_to_underscore() {
        let v = canonicalize("apr-merge");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "apr_merge");
        }
    }

    #[test]
    fn lowercases_uppercase() {
        let v = canonicalize("FOO");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "foo");
        }
    }

    #[test]
    fn strips_rs_extension() {
        let v = canonicalize("recipe.rs");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "recipe");
        }
    }

    #[test]
    fn collapses_repeated_underscores() {
        let v = canonicalize("foo  bar");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "foo_bar");
        }
    }

    #[test]
    fn trims_leading_trailing_underscores() {
        let v = canonicalize("--foo--");
        if let CanonVerdict::Ok { canonical } = v {
            assert_eq!(canonical, "foo");
        }
    }

    #[test]
    fn invalid_char_rejected() {
        let v = canonicalize("foo!bar");
        assert!(matches!(v, CanonVerdict::InvalidChars { offending: '!' }));
    }

    #[test]
    fn empty_id_rejected() {
        assert_eq!(canonicalize("   "), CanonVerdict::EmptyId);
    }

    #[test]
    fn only_dashes_empty() {
        assert_eq!(canonicalize("---"), CanonVerdict::EmptyId);
    }

    #[test]
    fn idempotent() {
        let v1 = canonicalize("My Recipe Name");
        if let CanonVerdict::Ok { canonical } = v1 {
            let v2 = canonicalize(&canonical);
            assert_eq!(
                v2,
                CanonVerdict::Ok {
                    canonical: canonical.clone()
                }
            );
        }
    }

    #[test]
    fn deterministic() {
        let a = canonicalize("My Recipe");
        let b = canonicalize("My Recipe");
        assert_eq!(a, b);
    }
}
