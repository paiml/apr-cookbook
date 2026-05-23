//! # Contracts-Macros Recipe Changelog Version
//!
//! Validate that the changelog header version matches the declared
//! crate version. Returns categorical verdict.
//!
//! Demonstrates the **CMM.182** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: keep-a-changelog format §2.1; conventional-changelog
//!  semver-bump rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_changelog_versi
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChangelogVersionVerdict {
    Match,
    Drift {
        crate_version: String,
        changelog_version: String,
    },
    InvalidConfig,
}

pub fn check(crate_version: &str, changelog_version: &str) -> ChangelogVersionVerdict {
    if crate_version.is_empty() || changelog_version.is_empty() {
        return ChangelogVersionVerdict::InvalidConfig;
    }
    if !is_valid_semver(crate_version) || !is_valid_semver(changelog_version) {
        return ChangelogVersionVerdict::InvalidConfig;
    }
    if crate_version == changelog_version {
        ChangelogVersionVerdict::Match
    } else {
        ChangelogVersionVerdict::Drift {
            crate_version: crate_version.to_string(),
            changelog_version: changelog_version.to_string(),
        }
    }
}

fn is_valid_semver(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    parts.iter().all(|p| p.parse::<u32>().is_ok())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_changelog_versi")?;

    println!("match: {:?}", check("1.2.3", "1.2.3"));
    println!("drift: {:?}", check("1.2.3", "1.2.0"));
    println!("invalid: {:?}", check("not-semver", "1.2.3"));
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
    fn matching_versions() {
        assert_eq!(check("1.2.3", "1.2.3"), ChangelogVersionVerdict::Match);
    }

    #[test]
    fn version_drift() {
        let v = check("1.2.3", "1.2.0");
        assert!(matches!(v, ChangelogVersionVerdict::Drift { .. }));
    }

    #[test]
    fn empty_crate_version_rejected() {
        assert_eq!(check("", "1.2.3"), ChangelogVersionVerdict::InvalidConfig);
    }

    #[test]
    fn empty_changelog_version_rejected() {
        assert_eq!(check("1.2.3", ""), ChangelogVersionVerdict::InvalidConfig);
    }

    #[test]
    fn non_semver_rejected() {
        assert_eq!(
            check("not-semver", "1.2.3"),
            ChangelogVersionVerdict::InvalidConfig
        );
    }

    #[test]
    fn missing_patch_rejected() {
        assert_eq!(
            check("1.2", "1.2.3"),
            ChangelogVersionVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let r1 = check("1.2.3", "1.2.3");
        let r2 = check("1.2.3", "1.2.3");
        assert_eq!(r1, r2);
    }

    #[test]
    fn drift_includes_both_versions() {
        let v = check("2.0.0", "1.0.0");
        if let ChangelogVersionVerdict::Drift {
            crate_version,
            changelog_version,
        } = v
        {
            assert_eq!(crate_version, "2.0.0");
            assert_eq!(changelog_version, "1.0.0");
        }
    }

    #[test]
    fn major_version_drift_detected() {
        assert!(matches!(
            check("2.0.0", "1.0.0"),
            ChangelogVersionVerdict::Drift { .. }
        ));
    }

    #[test]
    fn patch_version_drift_detected() {
        assert!(matches!(
            check("1.0.5", "1.0.4"),
            ChangelogVersionVerdict::Drift { .. }
        ));
    }

    #[test]
    fn zero_version_handled() {
        assert_eq!(check("0.0.0", "0.0.0"), ChangelogVersionVerdict::Match);
    }

    #[test]
    fn high_version_numbers_handled() {
        assert_eq!(
            check("100.200.300", "100.200.300"),
            ChangelogVersionVerdict::Match
        );
    }
}
