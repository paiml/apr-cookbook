//! # apr pull — `--revision` Pin Resolver
//!
//! `apr pull <REPO> --revision <REV>` pins to a branch, tag, or git SHA
//! per CRUX-A-03. Revisions can be: branch name (`main`), tag (`v1.0`),
//! short SHA (`abc1234`), or full 40-char SHA. This recipe builds the
//! revision classifier and asserts the contract: defaults to "main"
//! when omitted, validates SHA format, rejects empty.
//!
//! Demonstrates the **PULL.6** recipe for PMAT-101 (apr pull coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-A-03 + Hugging Face Hub revision API
//!
//! Run with: cargo run --example cli_pull_revision_pin_resolver
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionKind {
    BranchOrTag(String),
    ShortSha(String), // 7-39 hex chars
    FullSha(String),  // exactly 40 hex chars
}

#[derive(Debug, PartialEq)]
pub enum RevisionVerdict {
    Resolved(RevisionKind),
    Empty,
    InvalidSha { observed: String },
}

const DEFAULT_REVISION: &str = "main";

pub fn resolve_revision(rev: Option<&str>) -> RevisionVerdict {
    let raw = rev.unwrap_or(DEFAULT_REVISION).trim();
    if raw.is_empty() {
        return RevisionVerdict::Empty;
    }
    // Detect SHA-like values: all-hex.
    let all_hex = raw.chars().all(|c| c.is_ascii_hexdigit());
    if all_hex {
        match raw.len() {
            40 => RevisionVerdict::Resolved(RevisionKind::FullSha(raw.into())),
            7..=39 => RevisionVerdict::Resolved(RevisionKind::ShortSha(raw.into())),
            _ => RevisionVerdict::InvalidSha {
                observed: raw.into(),
            },
        }
    } else {
        RevisionVerdict::Resolved(RevisionKind::BranchOrTag(raw.into()))
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pull_revision_pin_resolver")?;

    let cases = [
        None,
        Some("main"),
        Some("v1.0.3"),
        Some("dev-branch"),
        Some("abc1234"),
        Some("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"),
        Some("abcdef"), // 6 hex chars — too short
        Some(""),
    ];
    for c in cases {
        println!("--revision {c:>50?}  →  {:?}", resolve_revision(c));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_revision_defaults_to_main() {
        let v = resolve_revision(None);
        assert_eq!(
            v,
            RevisionVerdict::Resolved(RevisionKind::BranchOrTag("main".into()))
        );
    }

    #[test]
    fn semver_tag_classified_as_branch_or_tag() {
        let v = resolve_revision(Some("v1.0.3"));
        assert_eq!(
            v,
            RevisionVerdict::Resolved(RevisionKind::BranchOrTag("v1.0.3".into()))
        );
    }

    #[test]
    fn full_40_char_sha_classified() {
        let sha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
        let v = resolve_revision(Some(sha));
        assert_eq!(
            v,
            RevisionVerdict::Resolved(RevisionKind::FullSha(sha.into()))
        );
    }

    #[test]
    fn short_7_to_39_sha_classified() {
        let v = resolve_revision(Some("abc1234"));
        assert_eq!(
            v,
            RevisionVerdict::Resolved(RevisionKind::ShortSha("abc1234".into()))
        );
    }

    #[test]
    fn sha_below_7_chars_rejected() {
        // < 7 hex chars is too ambiguous; would match many commits in a large repo.
        let v = resolve_revision(Some("abcdef"));
        assert!(matches!(v, RevisionVerdict::InvalidSha { .. }));
    }

    #[test]
    fn sha_above_40_chars_rejected() {
        // Git SHAs are exactly 40 hex chars (SHA-1).
        let v = resolve_revision(Some(&"a".repeat(50)));
        assert!(matches!(v, RevisionVerdict::InvalidSha { .. }));
    }

    #[test]
    fn empty_rejected() {
        let v = resolve_revision(Some(""));
        assert_eq!(v, RevisionVerdict::Empty);
    }

    #[test]
    fn whitespace_only_rejected() {
        let v = resolve_revision(Some("   "));
        assert_eq!(v, RevisionVerdict::Empty);
    }
}
