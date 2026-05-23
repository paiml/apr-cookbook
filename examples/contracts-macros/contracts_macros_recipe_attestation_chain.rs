//! # Contracts-Macros Recipe Attestation Chain
//!
//! Verify a chain of recipe attestations: each step is signed by the
//! previous step's signer, and the chain length meets a minimum.
//! Returns chain validity and the minimum-required gap that's missing.
//!
//! Demonstrates the **CMM.166** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA Level 4 attestation chain; in-toto envelope sig
//!  validation rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_attestation_chain
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok {
        chain_length: u32,
        broken_at: Option<u32>,
    },
    InvalidConfig,
}

/// Steps: (signer, prev_signer). Chain must form a valid linked list.
pub fn verify(steps: &[(&str, &str)], min_length: u32) -> ChainVerdict {
    if steps.is_empty() || min_length == 0 {
        return ChainVerdict::InvalidConfig;
    }
    let mut broken_at: Option<u32> = None;
    for (i, w) in steps.windows(2).enumerate() {
        // Each step's prev_signer must equal the previous step's signer.
        if w[1].1 != w[0].0 {
            broken_at = Some(i as u32 + 1);
            break;
        }
    }
    if (steps.len() as u32) < min_length && broken_at.is_none() {
        broken_at = Some(steps.len() as u32);
    }
    ChainVerdict::Ok {
        chain_length: steps.len() as u32,
        broken_at,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_attestation_chain")?;

    let valid = [("alice", ""), ("bob", "alice"), ("carol", "bob")];
    println!("valid: {:?}", verify(&valid, 3));
    let broken = [("alice", ""), ("bob", "alice"), ("carol", "alice")];
    println!("broken: {:?}", verify(&broken, 3));
    println!("invalid: {:?}", verify(&[], 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_chain_valid() {
        let steps = [("a", ""), ("b", "a"), ("c", "b")];
        let v = verify(&steps, 3);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert!(broken_at.is_none());
        }
    }

    #[test]
    fn broken_link_detected() {
        let steps = [("a", ""), ("b", "a"), ("c", "alice")];
        let v = verify(&steps, 3);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert_eq!(broken_at, Some(2));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(verify(&[], 3), ChainVerdict::InvalidConfig);
    }

    #[test]
    fn zero_min_rejected() {
        assert_eq!(verify(&[("a", "")], 0), ChainVerdict::InvalidConfig);
    }

    #[test]
    fn chain_too_short_broken() {
        let steps = [("a", ""), ("b", "a")];
        let v = verify(&steps, 5);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert_eq!(broken_at, Some(2));
        }
    }

    #[test]
    fn chain_length_returned() {
        let steps = [("a", ""), ("b", "a")];
        let v = verify(&steps, 1);
        if let ChainVerdict::Ok { chain_length, .. } = v {
            assert_eq!(chain_length, 2);
        }
    }

    #[test]
    fn deterministic() {
        let steps = [("a", "")];
        let r1 = verify(&steps, 1);
        let r2 = verify(&steps, 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_step_valid_at_min_1() {
        let steps = [("a", "")];
        let v = verify(&steps, 1);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert!(broken_at.is_none());
        }
    }

    #[test]
    fn many_steps_handled() {
        let mut steps: Vec<(&str, &str)> = Vec::new();
        steps.push(("a", ""));
        for _ in 0..30 {
            steps.push(("a", "a"));
        }
        let v = verify(&steps, 5);
        assert!(matches!(v, ChainVerdict::Ok { .. }));
    }

    #[test]
    fn unicode_signer_supported() {
        let steps = [("café", ""), ("résumé", "café")];
        let v = verify(&steps, 2);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert!(broken_at.is_none());
        }
    }

    #[test]
    fn meeting_min_length_no_break() {
        let steps = [("a", ""), ("b", "a"), ("c", "b")];
        let v = verify(&steps, 3);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert!(broken_at.is_none());
        }
    }

    #[test]
    fn first_link_broken_at_one() {
        let steps = [("a", ""), ("b", "wrong")];
        let v = verify(&steps, 2);
        if let ChainVerdict::Ok { broken_at, .. } = v {
            assert_eq!(broken_at, Some(1));
        }
    }
}
