//! # Registry Immutable Tag Policy
//!
//! Semantic tags (e.g., `1.2.3`, `latest`) have different mutability
//! rules:
//!   - exact-version (1.2.3) → IMMUTABLE; reject re-publish
//!   - rolling (latest, dev, nightly) → mutable; allow overwrite
//!   - branch (feature-x) → mutable; allow overwrite
//!   - sha-pinned (sha256:abc) → IMMUTABLE; never overwrite
//!
//! This recipe builds the policy checker.
//!
//! Demonstrates the **REG.15** recipe for PMAT-143 (registry round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace + Docker tag-immutability spec.
//!
//! Run with: cargo run --example registry_immutable_tag_policy
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Immutable,
    Rolling,
}

#[derive(Debug, PartialEq)]
pub enum PolicyVerdict {
    AllowPublish { mutability: Mutability },
    RejectImmutableExists { tag: String },
    EmptyTag,
    InvalidShaFormat,
}

const ROLLING_TAGS: &[&str] = &["latest", "dev", "nightly", "main", "master", "stable"];

pub fn check(tag: &str, exists: bool) -> PolicyVerdict {
    if tag.is_empty() {
        return PolicyVerdict::EmptyTag;
    }
    if let Some(rest) = tag.strip_prefix("sha256:") {
        if rest.len() != 64 || !rest.chars().all(|c| c.is_ascii_hexdigit()) {
            return PolicyVerdict::InvalidShaFormat;
        }
        if exists {
            return PolicyVerdict::RejectImmutableExists {
                tag: tag.to_string(),
            };
        }
        return PolicyVerdict::AllowPublish {
            mutability: Mutability::Immutable,
        };
    }
    if ROLLING_TAGS.contains(&tag) {
        return PolicyVerdict::AllowPublish {
            mutability: Mutability::Rolling,
        };
    }
    if is_semver(tag) {
        if exists {
            return PolicyVerdict::RejectImmutableExists {
                tag: tag.to_string(),
            };
        }
        return PolicyVerdict::AllowPublish {
            mutability: Mutability::Immutable,
        };
    }
    PolicyVerdict::AllowPublish {
        mutability: Mutability::Rolling,
    }
}

fn is_semver(tag: &str) -> bool {
    let parts: Vec<&str> = tag.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_immutable_tag_policy")?;

    println!("new 1.2.3: {:?}", check("1.2.3", false));
    println!("existing 1.2.3: {:?}", check("1.2.3", true));
    println!("existing latest: {:?}", check("latest", true));
    let sha = format!("sha256:{}", "a".repeat(64));
    println!("new sha-pin: {:?}", check(&sha, false));
    println!("invalid sha: {:?}", check("sha256:abc", false));
    println!("empty: {:?}", check("", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn new_semver_allowed() {
        let v = check("1.2.3", false);
        assert_eq!(
            v,
            PolicyVerdict::AllowPublish {
                mutability: Mutability::Immutable
            }
        );
    }

    #[test]
    fn existing_semver_rejected() {
        let v = check("1.2.3", true);
        assert!(matches!(v, PolicyVerdict::RejectImmutableExists { .. }));
    }

    #[test]
    fn rolling_tag_always_allowed() {
        for tag in ["latest", "dev", "nightly"] {
            let v = check(tag, true);
            assert_eq!(
                v,
                PolicyVerdict::AllowPublish {
                    mutability: Mutability::Rolling
                }
            );
        }
    }

    #[test]
    fn sha_pin_immutable_when_new() {
        let sha = format!("sha256:{}", "a".repeat(64));
        let v = check(&sha, false);
        assert_eq!(
            v,
            PolicyVerdict::AllowPublish {
                mutability: Mutability::Immutable
            }
        );
    }

    #[test]
    fn sha_pin_rejected_when_exists() {
        let sha = format!("sha256:{}", "a".repeat(64));
        let v = check(&sha, true);
        assert!(matches!(v, PolicyVerdict::RejectImmutableExists { .. }));
    }

    #[test]
    fn invalid_sha_rejected() {
        let v = check("sha256:abc", false);
        assert_eq!(v, PolicyVerdict::InvalidShaFormat);
    }

    #[test]
    fn non_hex_sha_rejected() {
        let bad = format!("sha256:{}", "z".repeat(64));
        assert_eq!(check(&bad, false), PolicyVerdict::InvalidShaFormat);
    }

    #[test]
    fn empty_tag_rejected() {
        assert_eq!(check("", false), PolicyVerdict::EmptyTag);
    }

    #[test]
    fn branch_tag_treated_as_rolling() {
        let v = check("feature-foo", true);
        assert_eq!(
            v,
            PolicyVerdict::AllowPublish {
                mutability: Mutability::Rolling
            }
        );
    }

    #[test]
    fn semver_with_letters_treated_as_rolling() {
        // "1.2.3-rc1" is not pure semver per this strict checker.
        let v = check("1.2.3-rc1", false);
        assert_eq!(
            v,
            PolicyVerdict::AllowPublish {
                mutability: Mutability::Rolling
            }
        );
    }
}
