//! # Contracts-Macros Spec Release Notes Completeness
//!
//! Validate spec release notes contain required sections (Added,
//! Changed, Fixed, Removed). Returns sorted missing-section names.
//!
//! Demonstrates the **CMM.183** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: keep-a-changelog format §3 standard sections; semver
//!  release-notes guidelines.
//!
//! Run with: cargo run --example contracts_macros_spec_release_notes
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ReleaseNotesVerdict {
    Ok {
        missing_sections: Vec<String>,
        present_sections: u32,
    },
    InvalidConfig,
}

pub fn check(present_sections: &[&str], required_sections: &[&str]) -> ReleaseNotesVerdict {
    if required_sections.is_empty() {
        return ReleaseNotesVerdict::InvalidConfig;
    }
    let present: BTreeSet<&str> = present_sections.iter().copied().collect();
    let missing: BTreeSet<String> = required_sections
        .iter()
        .filter(|s| !present.contains(*s))
        .map(|s| (*s).to_string())
        .collect();
    ReleaseNotesVerdict::Ok {
        missing_sections: missing.into_iter().collect(),
        present_sections: present.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_spec_release_notes")?;

    let required = ["Added", "Changed", "Fixed", "Removed"];
    let present = ["Added", "Fixed"];
    println!("check: {:?}", check(&present, &required));
    println!("invalid: {:?}", check(&[], &[]));
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
    fn all_present_no_missing() {
        let v = check(&["Added", "Changed"], &["Added", "Changed"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }

    #[test]
    fn missing_section_flagged() {
        let v = check(&["Added"], &["Added", "Changed"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections, vec!["Changed".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(check(&["Added"], &[]), ReleaseNotesVerdict::InvalidConfig);
    }

    #[test]
    fn empty_present_all_missing() {
        let v = check(&[], &["Added", "Changed"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections.len(), 2);
        }
    }

    #[test]
    fn extra_sections_ok() {
        let v = check(&["Added", "Changed", "Extra"], &["Added"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["Added"], &["Added"]);
        let r2 = check(&["Added"], &["Added"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn missing_sorted() {
        let v = check(&[], &["Removed", "Changed", "Added", "Fixed"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(
                missing_sections,
                vec![
                    "Added".to_string(),
                    "Changed".to_string(),
                    "Fixed".to_string(),
                    "Removed".to_string(),
                ]
            );
        }
    }

    #[test]
    fn present_count_correct() {
        let v = check(&["Added", "Changed"], &["Added"]);
        if let ReleaseNotesVerdict::Ok {
            present_sections, ..
        } = v
        {
            assert_eq!(present_sections, 2);
        }
    }

    #[test]
    fn duplicate_present_dedup() {
        let v = check(&["Added", "Added"], &["Added"]);
        if let ReleaseNotesVerdict::Ok {
            present_sections, ..
        } = v
        {
            assert_eq!(present_sections, 1);
        }
    }

    #[test]
    fn case_sensitive_section() {
        let v = check(&["added"], &["Added"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections, vec!["Added".to_string()]);
        }
    }

    #[test]
    fn many_required_handled() {
        let required: Vec<&str> = vec![
            "Added",
            "Changed",
            "Deprecated",
            "Removed",
            "Fixed",
            "Security",
        ];
        let v = check(&["Added"], &required);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert_eq!(missing_sections.len(), 5);
        }
    }

    #[test]
    fn unicode_section_supported() {
        let v = check(&["café"], &["café"]);
        if let ReleaseNotesVerdict::Ok {
            missing_sections, ..
        } = v
        {
            assert!(missing_sections.is_empty());
        }
    }
}
