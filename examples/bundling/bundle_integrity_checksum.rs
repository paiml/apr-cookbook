//! # Bundle Integrity Checksum Policy
//!
//! End-of-archive sha256 covers the manifest + tensor blob. Integrity
//! check timing: BeforeLoad (always — strict), OnDemand (per-tensor on
//! first access — fast startup), Skipped (off — only for trusted
//! local). This recipe builds the policy picker + hash classifier
//! (Match/Mismatch/MissingChecksum).
//!
//! Demonstrates the **BUNDLE.16** recipe for PMAT-133 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST FIPS 180-4 (SHA-2 family).
//!
//! Run with: cargo run --example bundle_integrity_checksum
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckPolicy {
    BeforeLoad,
    OnDemand,
    Skipped,
}

#[derive(Debug, PartialEq)]
pub enum PolicyVerdict {
    Ok(CheckPolicy),
    InvalidProvenance,
}

pub fn pick_policy(is_remote: bool, latency_critical: bool, trusted_path: bool) -> PolicyVerdict {
    if is_remote && trusted_path {
        return PolicyVerdict::InvalidProvenance;
    }
    let policy = if is_remote {
        CheckPolicy::BeforeLoad
    } else if latency_critical {
        CheckPolicy::OnDemand
    } else if trusted_path {
        CheckPolicy::Skipped
    } else {
        CheckPolicy::BeforeLoad
    };
    PolicyVerdict::Ok(policy)
}

#[derive(Debug, PartialEq)]
pub enum ChecksumVerdict {
    Match,
    Mismatch { expected: String, got: String },
    MissingExpected,
    InvalidHashFormat { len: usize },
}

const SHA256_HEX_LEN: usize = 64;

pub fn verify(expected_hex: Option<&str>, computed_hex: &str) -> ChecksumVerdict {
    if computed_hex.len() != SHA256_HEX_LEN || !computed_hex.chars().all(|c| c.is_ascii_hexdigit())
    {
        return ChecksumVerdict::InvalidHashFormat {
            len: computed_hex.len(),
        };
    }
    let Some(expected) = expected_hex else {
        return ChecksumVerdict::MissingExpected;
    };
    if expected.len() != SHA256_HEX_LEN {
        return ChecksumVerdict::InvalidHashFormat {
            len: expected.len(),
        };
    }
    if expected.eq_ignore_ascii_case(computed_hex) {
        ChecksumVerdict::Match
    } else {
        ChecksumVerdict::Mismatch {
            expected: expected.into(),
            got: computed_hex.into(),
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_integrity_checksum")?;

    for (remote, latency, trusted) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (false, false, false),
        (true, false, true),
    ] {
        println!(
            "remote={remote} latency={latency} trusted={trusted}  →  {:?}",
            pick_policy(remote, latency, trusted)
        );
    }

    let h = "a".repeat(64);
    println!("match: {:?}", verify(Some(&h), &h));
    println!(
        "mismatch: {:?}",
        verify(Some(&"a".repeat(64)), &"b".repeat(64))
    );
    println!("bad len: {:?}", verify(Some(&h), "abc"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn remote_picks_before_load() {
        assert_eq!(
            pick_policy(true, false, false),
            PolicyVerdict::Ok(CheckPolicy::BeforeLoad)
        );
    }

    #[test]
    fn local_latency_picks_on_demand() {
        assert_eq!(
            pick_policy(false, true, false),
            PolicyVerdict::Ok(CheckPolicy::OnDemand)
        );
    }

    #[test]
    fn local_trusted_picks_skipped() {
        assert_eq!(
            pick_policy(false, false, true),
            PolicyVerdict::Ok(CheckPolicy::Skipped)
        );
    }

    #[test]
    fn local_default_picks_before_load() {
        assert_eq!(
            pick_policy(false, false, false),
            PolicyVerdict::Ok(CheckPolicy::BeforeLoad)
        );
    }

    #[test]
    fn remote_trusted_invalid_provenance() {
        // Cannot be both remote AND trusted-path.
        assert_eq!(
            pick_policy(true, false, true),
            PolicyVerdict::InvalidProvenance
        );
    }

    #[test]
    fn matching_hash_verifies() {
        let h = "a".repeat(64);
        assert_eq!(verify(Some(&h), &h), ChecksumVerdict::Match);
    }

    #[test]
    fn mismatched_hash_detected() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let v = verify(Some(&a), &b);
        assert!(matches!(v, ChecksumVerdict::Mismatch { .. }));
    }

    #[test]
    fn missing_expected_returned() {
        let h = "a".repeat(64);
        assert_eq!(verify(None, &h), ChecksumVerdict::MissingExpected);
    }

    #[test]
    fn case_insensitive_match() {
        let lower = "a".repeat(64);
        let upper = "A".repeat(64);
        assert_eq!(verify(Some(&lower), &upper), ChecksumVerdict::Match);
    }

    #[test]
    fn short_computed_invalid() {
        let h = "a".repeat(64);
        let v = verify(Some(&h), "abc");
        assert!(matches!(v, ChecksumVerdict::InvalidHashFormat { len: 3 }));
    }

    #[test]
    fn non_hex_computed_invalid() {
        let h = "a".repeat(64);
        let bad = "z".repeat(64);
        let v = verify(Some(&h), &bad);
        assert!(matches!(v, ChecksumVerdict::InvalidHashFormat { .. }));
    }
}
