//! # Bundle Signing Chain Builder
//!
//! APR signed bundles include a chain of signatures: model bytes →
//! sha256 → ed25519(sha256, key_a) → ed25519(prev_sig + ts, key_b).
//! Verification walks the chain from root to leaf, checking each link.
//! This recipe builds the chain validator + ordering check.
//!
//! Demonstrates the **BUNDLE.13** recipe for PMAT-127 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ENC-001 + RFC 8032 (Ed25519).
//!
//! Run with: cargo run --example bundle_signing_chain_builder
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainLink {
    pub signer: String,
    pub timestamp_ms: u64,
    pub signature_hex: String,
}

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok,
    EmptyChain,
    NonMonotonicTimestamps { at_index: usize },
    DuplicateSigner { signer: String },
    InvalidSignatureLength { at_index: usize, len: usize },
}

const ED25519_HEX_LEN: usize = 128;

pub fn validate(chain: &[ChainLink]) -> ChainVerdict {
    if chain.is_empty() {
        return ChainVerdict::EmptyChain;
    }
    let mut seen_signers: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut prev_ts = 0u64;
    for (i, link) in chain.iter().enumerate() {
        if link.signature_hex.len() != ED25519_HEX_LEN {
            return ChainVerdict::InvalidSignatureLength {
                at_index: i,
                len: link.signature_hex.len(),
            };
        }
        if !seen_signers.insert(link.signer.as_str()) {
            return ChainVerdict::DuplicateSigner {
                signer: link.signer.clone(),
            };
        }
        if i > 0 && link.timestamp_ms < prev_ts {
            return ChainVerdict::NonMonotonicTimestamps { at_index: i };
        }
        prev_ts = link.timestamp_ms;
    }
    ChainVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_signing_chain_builder")?;

    let valid = vec![
        ChainLink {
            signer: "build-bot".into(),
            timestamp_ms: 1_700_000_000_000,
            signature_hex: "a".repeat(128),
        },
        ChainLink {
            signer: "release-mgr".into(),
            timestamp_ms: 1_700_000_001_000,
            signature_hex: "b".repeat(128),
        },
    ];
    println!("valid: {:?}", validate(&valid));

    let dup = vec![
        ChainLink {
            signer: "x".into(),
            timestamp_ms: 1,
            signature_hex: "a".repeat(128),
        },
        ChainLink {
            signer: "x".into(),
            timestamp_ms: 2,
            signature_hex: "b".repeat(128),
        },
    ];
    println!("dup: {:?}", validate(&dup));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(signer: &str, ts: u64) -> ChainLink {
        ChainLink {
            signer: signer.into(),
            timestamp_ms: ts,
            signature_hex: "a".repeat(128),
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_chain_passes() {
        let chain = vec![link("a", 100), link("b", 200), link("c", 300)];
        assert_eq!(validate(&chain), ChainVerdict::Ok);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[]), ChainVerdict::EmptyChain);
    }

    #[test]
    fn equal_timestamps_passes() {
        // Non-monotonic = strictly less than; equal is OK (concurrent signing).
        let chain = vec![link("a", 100), link("b", 100)];
        assert_eq!(validate(&chain), ChainVerdict::Ok);
    }

    #[test]
    fn monotonic_violation_rejected() {
        let chain = vec![link("a", 200), link("b", 100)];
        let v = validate(&chain);
        assert!(matches!(
            v,
            ChainVerdict::NonMonotonicTimestamps { at_index: 1 }
        ));
    }

    #[test]
    fn duplicate_signer_rejected() {
        let chain = vec![link("a", 100), link("a", 200)];
        let v = validate(&chain);
        assert!(matches!(v, ChainVerdict::DuplicateSigner { .. }));
    }

    #[test]
    fn invalid_signature_length_rejected() {
        let mut chain = vec![link("a", 100)];
        chain[0].signature_hex = "abc".into();
        let v = validate(&chain);
        assert!(matches!(
            v,
            ChainVerdict::InvalidSignatureLength {
                at_index: 0,
                len: 3
            }
        ));
    }

    #[test]
    fn first_link_signature_validated() {
        let mut chain = vec![link("a", 100), link("b", 200)];
        chain[0].signature_hex = "tooshort".into();
        let v = validate(&chain);
        assert!(matches!(
            v,
            ChainVerdict::InvalidSignatureLength { at_index: 0, .. }
        ));
    }

    #[test]
    fn single_signer_passes() {
        assert_eq!(validate(&[link("solo", 100)]), ChainVerdict::Ok);
    }

    #[test]
    fn long_chain_with_unique_signers_passes() {
        let chain: Vec<ChainLink> = (0..10u64)
            .map(|i| link(&format!("signer-{i}"), i * 100))
            .collect();
        assert_eq!(validate(&chain), ChainVerdict::Ok);
    }
}
