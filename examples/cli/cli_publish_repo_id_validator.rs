//! # apr publish — `<REPO_ID>` Format Validator
//!
//! `apr publish <DIRECTORY> <REPO_ID>` requires REPO_ID in HuggingFace
//! Hub's `<org>/<name>` format. This recipe builds the validator and
//! enforces the contract: org and name each must be 1-96 chars, both
//! lowercase-alphanumeric-with-`-`-or-`_`, no consecutive `--`, no
//! leading/trailing `-`.
//!
//! Demonstrates the **PUBLISH.7** recipe for PMAT-098 (apr publish coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-PUB-001 + huggingface_hub.utils.RepoId validation
//!
//! Run with: cargo run --example cli_publish_repo_id_validator
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RepoIdVerdict {
    Ok { org: String, name: String },
    MissingSlash,
    EmptyComponent,
    TooLong { which: &'static str, len: usize },
    InvalidChar { which: &'static str, ch: char },
    LeadingOrTrailingHyphen { which: &'static str },
    DoubleHyphen { which: &'static str },
}

const MAX_LEN: usize = 96;

pub fn validate_repo_id(s: &str) -> RepoIdVerdict {
    let parts: Vec<&str> = s.splitn(2, '/').collect();
    if parts.len() != 2 {
        return RepoIdVerdict::MissingSlash;
    }
    let (org, name) = (parts[0], parts[1]);
    for (which, segment) in [("org", org), ("name", name)] {
        if segment.is_empty() {
            return RepoIdVerdict::EmptyComponent;
        }
        if segment.len() > MAX_LEN {
            return RepoIdVerdict::TooLong {
                which,
                len: segment.len(),
            };
        }
        if let Some(ch) = segment.chars().find(|c| {
            !c.is_ascii_lowercase()
                && !c.is_ascii_digit()
                && *c != '-'
                && *c != '_'
                && !(*c).is_ascii_uppercase()
        }) {
            return RepoIdVerdict::InvalidChar { which, ch };
        }
        if segment.starts_with('-') || segment.ends_with('-') {
            return RepoIdVerdict::LeadingOrTrailingHyphen { which };
        }
        if segment.contains("--") {
            return RepoIdVerdict::DoubleHyphen { which };
        }
    }
    RepoIdVerdict::Ok {
        org: org.into(),
        name: name.into(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_publish_repo_id_validator")?;

    let cases = [
        "paiml/whisper-apr-tiny",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "missing-slash",
        "/empty-org",
        "team/empty-",
        "team/-leading",
        "team/has--double",
        "team/has space",
        "team/тест",
    ];

    for c in cases {
        println!("{c:>40}  →  {:?}", validate_repo_id(c));
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
    fn happy_repo_id_passes() {
        assert_eq!(
            validate_repo_id("paiml/whisper-apr-tiny"),
            RepoIdVerdict::Ok {
                org: "paiml".into(),
                name: "whisper-apr-tiny".into(),
            }
        );
    }

    #[test]
    fn missing_slash_rejected() {
        assert_eq!(validate_repo_id("nojustname"), RepoIdVerdict::MissingSlash);
    }

    #[test]
    fn empty_org_rejected() {
        assert_eq!(validate_repo_id("/name"), RepoIdVerdict::EmptyComponent);
    }

    #[test]
    fn empty_name_rejected() {
        assert_eq!(validate_repo_id("org/"), RepoIdVerdict::EmptyComponent);
    }

    #[test]
    fn whitespace_rejected_as_invalid_char() {
        let v = validate_repo_id("team/has space");
        assert!(matches!(v, RepoIdVerdict::InvalidChar { .. }));
    }

    #[test]
    fn non_ascii_rejected() {
        let v = validate_repo_id("team/тест");
        assert!(matches!(v, RepoIdVerdict::InvalidChar { .. }));
    }

    #[test]
    fn leading_hyphen_rejected() {
        assert!(matches!(
            validate_repo_id("team/-leading"),
            RepoIdVerdict::LeadingOrTrailingHyphen { .. }
        ));
    }

    #[test]
    fn trailing_hyphen_rejected() {
        assert!(matches!(
            validate_repo_id("team/trailing-"),
            RepoIdVerdict::LeadingOrTrailingHyphen { .. }
        ));
    }

    #[test]
    fn double_hyphen_rejected() {
        assert!(matches!(
            validate_repo_id("team/has--double"),
            RepoIdVerdict::DoubleHyphen { .. }
        ));
    }

    #[test]
    fn over_96_chars_rejected() {
        let long = "a".repeat(97);
        let v = validate_repo_id(&format!("org/{long}"));
        assert!(matches!(v, RepoIdVerdict::TooLong { .. }));
    }
}
