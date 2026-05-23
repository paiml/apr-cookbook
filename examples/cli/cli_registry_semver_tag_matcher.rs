//! # apr registry — Semver Tag Matcher
//!
//! Tags follow semver: `MAJOR.MINOR.PATCH[-PRERELEASE]`. Constraints:
//! `^X.Y.Z` (caret, compatible), `~X.Y.Z` (tilde, patch-only), `=X.Y.Z`
//! (exact). Pre-release tags are excluded from caret/tilde matches.
//! This recipe builds the matcher.
//!
//! Demonstrates the **REG.6** recipe for PMAT-114 (apr registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender REG-001 + Semver 2.0.0 spec
//!
//! Run with: cargo run --example cli_registry_semver_tag_matcher
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
}

pub fn parse_version(s: &str) -> Option<Version> {
    let (core, prerelease) = match s.split_once('-') {
        Some((c, p)) => (c, Some(p.to_string())),
        None => (s, None),
    };
    let parts: Vec<&str> = core.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some(Version {
        major: parts[0].parse().ok()?,
        minor: parts[1].parse().ok()?,
        patch: parts[2].parse().ok()?,
        prerelease,
    })
}

#[derive(Debug)]
pub enum Constraint {
    Exact(Version),
    Caret(Version),
    Tilde(Version),
}

pub fn parse_constraint(s: &str) -> Option<Constraint> {
    if let Some(rest) = s.strip_prefix('^') {
        Some(Constraint::Caret(parse_version(rest)?))
    } else if let Some(rest) = s.strip_prefix('~') {
        Some(Constraint::Tilde(parse_version(rest)?))
    } else if let Some(rest) = s.strip_prefix('=') {
        Some(Constraint::Exact(parse_version(rest)?))
    } else {
        Some(Constraint::Exact(parse_version(s)?))
    }
}

pub fn matches(constraint: &Constraint, candidate: &Version) -> bool {
    match constraint {
        Constraint::Exact(v) => v == candidate,
        Constraint::Caret(v) => {
            // Same major, ≥ minor.patch, no prerelease.
            candidate.prerelease.is_none()
                && candidate.major == v.major
                && (candidate.minor > v.minor
                    || (candidate.minor == v.minor && candidate.patch >= v.patch))
        }
        Constraint::Tilde(v) => {
            // Same major + minor, ≥ patch, no prerelease.
            candidate.prerelease.is_none()
                && candidate.major == v.major
                && candidate.minor == v.minor
                && candidate.patch >= v.patch
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_semver_tag_matcher")?;

    let cases = [
        ("^1.2.3", "1.5.0"),
        ("^1.2.3", "2.0.0"),
        ("~1.2.3", "1.2.7"),
        ("~1.2.3", "1.3.0"),
        ("=1.2.3", "1.2.3"),
        ("^1.2.3", "1.5.0-rc1"),
    ];
    for (c, cand) in cases {
        let constraint = parse_constraint(c).unwrap();
        let candidate = parse_version(cand).unwrap();
        println!("{c} matches {cand}? {}", matches(&constraint, &candidate));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_matches_only_identical() {
        let c = parse_constraint("=1.2.3").unwrap();
        assert!(matches(&c, &parse_version("1.2.3").unwrap()));
        assert!(!matches(&c, &parse_version("1.2.4").unwrap()));
    }

    #[test]
    fn caret_allows_minor_bump() {
        let c = parse_constraint("^1.2.3").unwrap();
        assert!(matches(&c, &parse_version("1.5.0").unwrap()));
        assert!(matches(&c, &parse_version("1.99.99").unwrap()));
    }

    #[test]
    fn caret_blocks_major_bump() {
        let c = parse_constraint("^1.2.3").unwrap();
        assert!(!matches(&c, &parse_version("2.0.0").unwrap()));
    }

    #[test]
    fn caret_blocks_below_baseline() {
        let c = parse_constraint("^1.2.3").unwrap();
        assert!(!matches(&c, &parse_version("1.2.2").unwrap()));
        assert!(!matches(&c, &parse_version("1.1.99").unwrap()));
    }

    #[test]
    fn tilde_allows_patch_bump_only() {
        let c = parse_constraint("~1.2.3").unwrap();
        assert!(matches(&c, &parse_version("1.2.99").unwrap()));
        assert!(!matches(&c, &parse_version("1.3.0").unwrap()));
    }

    #[test]
    fn caret_excludes_prerelease() {
        let c = parse_constraint("^1.2.3").unwrap();
        // Pre-release versions should not match a stable constraint.
        assert!(!matches(&c, &parse_version("1.5.0-rc1").unwrap()));
    }

    #[test]
    fn parse_invalid_returns_none() {
        assert!(parse_version("1.2").is_none());
        assert!(parse_version("v1.2.3").is_none());
        assert!(parse_version("1.2.x").is_none());
    }

    #[test]
    fn bare_version_treated_as_exact() {
        let c = parse_constraint("1.2.3").unwrap();
        assert!(matches(&c, &parse_version("1.2.3").unwrap()));
        assert!(!matches(&c, &parse_version("1.2.4").unwrap()));
    }
}
