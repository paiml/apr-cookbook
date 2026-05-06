//! # Registry Signature Chain Verifier
//!
//! Models in a registry can be signed by multiple actors:
//! publisher → mirror → re-signer → user-trust-anchor. The chain is
//! valid iff every adjacent pair (signer_i, signer_{i+1}) shows up in
//! the trust-graph as an allowed delegation, AND the chain ends at a
//! root the user trusts.
//!
//! This recipe builds the verifier.
//!
//! Demonstrates the **REG.12** recipe for PMAT-138 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sigstore / TUF (The Update Framework) signature chain model.
//!
//! Run with: cargo run --example registry_signature_chain
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok { trusted_root: String },
    EmptyChain,
    UntrustedRoot { last_signer: String },
    InvalidDelegation { from: String, to: String },
}

pub fn verify(
    chain: &[&str],
    delegations: &BTreeSet<(String, String)>,
    trusted_roots: &BTreeSet<String>,
) -> ChainVerdict {
    if chain.is_empty() {
        return ChainVerdict::EmptyChain;
    }
    for w in chain.windows(2) {
        let pair = (w[0].to_string(), w[1].to_string());
        if !delegations.contains(&pair) {
            return ChainVerdict::InvalidDelegation {
                from: w[0].to_string(),
                to: w[1].to_string(),
            };
        }
    }
    let last = chain.last().unwrap();
    if !trusted_roots.contains(*last) {
        return ChainVerdict::UntrustedRoot {
            last_signer: (*last).to_string(),
        };
    }
    ChainVerdict::Ok {
        trusted_root: (*last).to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_signature_chain")?;

    let mut delegations = BTreeSet::new();
    delegations.insert(("publisher".to_string(), "mirror".to_string()));
    delegations.insert(("mirror".to_string(), "user-anchor".to_string()));

    let mut roots = BTreeSet::new();
    roots.insert("user-anchor".to_string());

    println!(
        "valid 3-step: {:?}",
        verify(
            &["publisher", "mirror", "user-anchor"],
            &delegations,
            &roots
        )
    );
    println!(
        "untrusted root: {:?}",
        verify(&["publisher", "mirror"], &delegations, &roots)
    );
    println!(
        "invalid delegation: {:?}",
        verify(&["publisher", "user-anchor"], &delegations, &roots)
    );
    println!("empty chain: {:?}", verify(&[], &delegations, &roots));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delegations(pairs: &[(&str, &str)]) -> BTreeSet<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
            .collect()
    }

    fn roots(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn verifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_valid_chain_accepts() {
        let dels = delegations(&[("a", "b"), ("b", "c")]);
        let rs = roots(&["c"]);
        let v = verify(&["a", "b", "c"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::Ok { .. }));
    }

    #[test]
    fn empty_chain_rejected() {
        let dels = delegations(&[]);
        let rs = roots(&["a"]);
        assert_eq!(verify(&[], &dels, &rs), ChainVerdict::EmptyChain);
    }

    #[test]
    fn untrusted_last_rejected() {
        let dels = delegations(&[("a", "b")]);
        let rs = roots(&["c"]);
        let v = verify(&["a", "b"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::UntrustedRoot { .. }));
    }

    #[test]
    fn missing_delegation_rejected() {
        let dels = delegations(&[("a", "b")]);
        let rs = roots(&["c"]);
        let v = verify(&["a", "c"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::InvalidDelegation { .. }));
    }

    #[test]
    fn single_node_must_be_root() {
        let dels = delegations(&[]);
        let rs = roots(&["a"]);
        // Chain of length 1 with a trusted root is fine.
        let v = verify(&["a"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::Ok { .. }));
    }

    #[test]
    fn single_node_not_root_rejected() {
        let dels = delegations(&[]);
        let rs = roots(&["b"]);
        let v = verify(&["a"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::UntrustedRoot { .. }));
    }

    #[test]
    fn long_chain_works() {
        let dels = delegations(&[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]);
        let rs = roots(&["e"]);
        let v = verify(&["a", "b", "c", "d", "e"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::Ok { .. }));
    }

    #[test]
    fn delegation_order_matters() {
        // (a, b) does not allow (b, a) chain.
        let dels = delegations(&[("a", "b")]);
        let rs = roots(&["a"]);
        let v = verify(&["b", "a"], &dels, &rs);
        assert!(matches!(v, ChainVerdict::InvalidDelegation { .. }));
    }

    #[test]
    fn multiple_trusted_roots_supported() {
        let dels = delegations(&[("a", "b"), ("a", "c")]);
        let rs = roots(&["b", "c"]);
        assert!(matches!(
            verify(&["a", "b"], &dels, &rs),
            ChainVerdict::Ok { .. }
        ));
        assert!(matches!(
            verify(&["a", "c"], &dels, &rs),
            ChainVerdict::Ok { .. }
        ));
    }

    #[test]
    fn first_invalid_delegation_reported() {
        let dels = delegations(&[("a", "b")]);
        let rs = roots(&["d"]);
        if let ChainVerdict::InvalidDelegation { from, to } = verify(&["a", "x", "d"], &dels, &rs) {
            assert_eq!(from, "a");
            assert_eq!(to, "x");
        }
    }
}
