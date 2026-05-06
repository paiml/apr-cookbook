//! # Contracts-Macros Alias Resolver
//!
//! Resolve a contract alias to its canonical name. Aliases are
//! case-insensitive and dash/underscore-equivalent. Returns the
//! canonical match or NotFound.
//!
//! Demonstrates the **CMM.35** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cargo manifest alias resolution.
//!
//! Run with: cargo run --example contracts_macros_alias_resolver
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AliasVerdict {
    Resolved { canonical: String },
    NotFound,
    EmptyQuery,
    AmbiguousMatches { count: u32 },
}

pub fn resolve(query: &str, canonical_names: &[&str]) -> AliasVerdict {
    let q = query.trim();
    if q.is_empty() {
        return AliasVerdict::EmptyQuery;
    }
    let normalized_q = normalize(q);
    let matches: Vec<&&str> = canonical_names
        .iter()
        .filter(|name| normalize(name) == normalized_q)
        .collect();
    match matches.len() {
        0 => AliasVerdict::NotFound,
        1 => AliasVerdict::Resolved {
            canonical: (*matches[0]).to_string(),
        },
        n => AliasVerdict::AmbiguousMatches { count: n as u32 },
    }
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c.to_ascii_lowercase())
            } else if c == '-' || c == '_' || c == ' ' {
                Some('_')
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_alias_resolver")?;

    let names = ["snake_case_name", "another_recipe"];
    println!("exact: {:?}", resolve("snake_case_name", &names));
    println!("kebab: {:?}", resolve("snake-case-name", &names));
    println!("title: {:?}", resolve("Snake Case Name", &names));
    println!("not found: {:?}", resolve("ghost", &names));
    println!("empty: {:?}", resolve("  ", &names));

    let dup = ["foo_bar", "foo-bar"];
    println!("ambiguous: {:?}", resolve("foo bar", &dup));
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
    fn exact_match() {
        let v = resolve("recipe", &["recipe", "other"]);
        if let AliasVerdict::Resolved { canonical } = v {
            assert_eq!(canonical, "recipe");
        }
    }

    #[test]
    fn dash_to_underscore() {
        let v = resolve("snake-case", &["snake_case"]);
        if let AliasVerdict::Resolved { canonical } = v {
            assert_eq!(canonical, "snake_case");
        }
    }

    #[test]
    fn space_to_underscore() {
        let v = resolve("snake case", &["snake_case"]);
        if let AliasVerdict::Resolved { canonical } = v {
            assert_eq!(canonical, "snake_case");
        }
    }

    #[test]
    fn case_insensitive() {
        let v = resolve("SnakeCase", &["snakecase"]);
        if let AliasVerdict::Resolved { canonical } = v {
            assert_eq!(canonical, "snakecase");
        }
    }

    #[test]
    fn not_found() {
        assert_eq!(resolve("ghost", &["foo", "bar"]), AliasVerdict::NotFound);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(resolve("  ", &["foo"]), AliasVerdict::EmptyQuery);
    }

    #[test]
    fn ambiguous_match() {
        let v = resolve("foo bar", &["foo_bar", "foo-bar"]);
        if let AliasVerdict::AmbiguousMatches { count } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn no_canonicals() {
        assert_eq!(resolve("any", &[]), AliasVerdict::NotFound);
    }

    #[test]
    fn special_chars_stripped() {
        let v = resolve("snake@case!", &["snakecase"]);
        if let AliasVerdict::Resolved { canonical } = v {
            assert_eq!(canonical, "snakecase");
        }
    }

    #[test]
    fn deterministic() {
        let names = ["snake_case", "other"];
        let a = resolve("snake-case", &names);
        let b = resolve("snake-case", &names);
        assert_eq!(a, b);
    }
}
