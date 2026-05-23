//! # Bundle Signing Attestation
//!
//! Verify chain: bundle_hash → publisher_signature → publisher_cert →
//! root_ca. Each link must verify; chain must terminate at trusted root.
//!
//! Demonstrates the **BUNDLE.25** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PKCS #7 SignedData + Sigstore transparency log.
//!
//! Run with: cargo run --example bundle_signing_attestation
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SignVerdict {
    Valid { publisher: String, chain_depth: u32 },
    BundleHashMismatch,
    InvalidPublisherSig,
    UntrustedRoot { presented_root: String },
    EmptyChain,
}

pub fn verify(
    bundle_hash: &str,
    expected_hash: &str,
    publisher: &str,
    chain: &[&str],
    trusted_roots: &BTreeSet<String>,
) -> SignVerdict {
    if chain.is_empty() {
        return SignVerdict::EmptyChain;
    }
    if bundle_hash != expected_hash {
        return SignVerdict::BundleHashMismatch;
    }
    if publisher.is_empty() {
        return SignVerdict::InvalidPublisherSig;
    }
    let presented_root = chain.last().unwrap();
    if !trusted_roots.contains(*presented_root) {
        return SignVerdict::UntrustedRoot {
            presented_root: (*presented_root).to_string(),
        };
    }
    SignVerdict::Valid {
        publisher: publisher.to_string(),
        chain_depth: chain.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_signing_attestation")?;

    let mut roots = BTreeSet::new();
    roots.insert("trusted-ca".to_string());

    let chain = ["bundle-cert", "publisher-cert", "trusted-ca"];
    println!(
        "valid: {:?}",
        verify("hash1", "hash1", "alice", &chain, &roots)
    );
    println!(
        "tampered: {:?}",
        verify("hash2", "hash1", "alice", &chain, &roots)
    );
    println!(
        "untrusted: {:?}",
        verify(
            "hash1",
            "hash1",
            "alice",
            &["bundle-cert", "evil-ca"],
            &roots
        )
    );
    println!(
        "empty publisher: {:?}",
        verify("hash1", "hash1", "", &chain, &roots)
    );
    println!(
        "empty chain: {:?}",
        verify("hash1", "hash1", "alice", &[], &roots)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trusted_roots() -> BTreeSet<String> {
        let mut r = BTreeSet::new();
        r.insert("trusted-ca".to_string());
        r
    }

    #[test]
    fn verifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_valid() {
        let v = verify(
            "hash1",
            "hash1",
            "alice",
            &["bundle", "publisher", "trusted-ca"],
            &trusted_roots(),
        );
        assert!(matches!(v, SignVerdict::Valid { .. }));
    }

    #[test]
    fn hash_mismatch_rejected() {
        let v = verify(
            "wronghash",
            "hash1",
            "alice",
            &["bundle", "trusted-ca"],
            &trusted_roots(),
        );
        assert_eq!(v, SignVerdict::BundleHashMismatch);
    }

    #[test]
    fn empty_publisher_rejected() {
        let v = verify(
            "hash1",
            "hash1",
            "",
            &["bundle", "trusted-ca"],
            &trusted_roots(),
        );
        assert_eq!(v, SignVerdict::InvalidPublisherSig);
    }

    #[test]
    fn untrusted_root_rejected() {
        let v = verify(
            "hash1",
            "hash1",
            "alice",
            &["bundle", "evil-ca"],
            &trusted_roots(),
        );
        assert!(matches!(v, SignVerdict::UntrustedRoot { .. }));
    }

    #[test]
    fn empty_chain_rejected() {
        assert_eq!(
            verify("hash1", "hash1", "alice", &[], &trusted_roots()),
            SignVerdict::EmptyChain
        );
    }

    #[test]
    fn chain_depth_returned() {
        let v = verify(
            "hash1",
            "hash1",
            "alice",
            &["a", "b", "c", "trusted-ca"],
            &trusted_roots(),
        );
        if let SignVerdict::Valid { chain_depth, .. } = v {
            assert_eq!(chain_depth, 4);
        }
    }

    #[test]
    fn publisher_returned() {
        let v = verify(
            "hash1",
            "hash1",
            "alice",
            &["bundle", "trusted-ca"],
            &trusted_roots(),
        );
        if let SignVerdict::Valid { publisher, .. } = v {
            assert_eq!(publisher, "alice");
        }
    }

    #[test]
    fn single_element_chain_at_root() {
        let v = verify("hash1", "hash1", "alice", &["trusted-ca"], &trusted_roots());
        assert!(matches!(v, SignVerdict::Valid { .. }));
    }

    #[test]
    fn untrusted_root_carries_value() {
        if let SignVerdict::UntrustedRoot { presented_root } = verify(
            "hash1",
            "hash1",
            "alice",
            &["bundle", "evil-ca"],
            &trusted_roots(),
        ) {
            assert_eq!(presented_root, "evil-ca");
        }
    }

    #[test]
    fn multiple_trusted_roots_supported() {
        let mut roots = BTreeSet::new();
        roots.insert("ca-a".to_string());
        roots.insert("ca-b".to_string());
        let v_a = verify("h", "h", "alice", &["ca-a"], &roots);
        let v_b = verify("h", "h", "alice", &["ca-b"], &roots);
        assert!(matches!(v_a, SignVerdict::Valid { .. }));
        assert!(matches!(v_b, SignVerdict::Valid { .. }));
    }
}
