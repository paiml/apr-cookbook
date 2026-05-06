//! # apr encrypt — Nonce Uniqueness Checker (AES-256-GCM)
//!
//! AES-GCM nonces MUST be unique per key. Reuse breaks confidentiality
//! AND integrity (Joux 2006: forbidden attack). This recipe checks a
//! batch of nonces for duplicates + length (96-bit / 12-byte standard).
//!
//! Demonstrates the **ENC.5** recipe for PMAT-115 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ENC-001 + Joux 2006 (forbidden attack on GCM)
//!
//! Run with: cargo run --example cli_encrypt_nonce_uniqueness_checker
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

const GCM_NONCE_BYTES: usize = 12;

#[derive(Debug, PartialEq)]
pub enum NonceVerdict {
    Ok,
    DuplicateAt { index: usize },
    WrongLength { index: usize, len: usize },
    Empty,
}

pub fn check(nonces: &[Vec<u8>]) -> NonceVerdict {
    if nonces.is_empty() {
        return NonceVerdict::Empty;
    }
    let mut seen: HashSet<&[u8]> = HashSet::new();
    for (i, n) in nonces.iter().enumerate() {
        if n.len() != GCM_NONCE_BYTES {
            return NonceVerdict::WrongLength {
                index: i,
                len: n.len(),
            };
        }
        if !seen.insert(n.as_slice()) {
            return NonceVerdict::DuplicateAt { index: i };
        }
    }
    NonceVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_nonce_uniqueness_checker")?;

    let unique = vec![vec![0u8; 12], vec![1u8; 12], vec![2u8; 12]];
    let dup = vec![vec![0u8; 12], vec![1u8; 12], vec![0u8; 12]];
    let bad_len = vec![vec![0u8; 8]];

    println!("unique → {:?}", check(&unique));
    println!("dup    → {:?}", check(&dup));
    println!("badlen → {:?}", check(&bad_len));
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
    fn unique_nonces_pass() {
        let nonces = vec![vec![0u8; 12], vec![1u8; 12], vec![2u8; 12]];
        assert_eq!(check(&nonces), NonceVerdict::Ok);
    }

    #[test]
    fn duplicate_at_index_detected() {
        let nonces = vec![vec![0u8; 12], vec![1u8; 12], vec![0u8; 12]];
        let v = check(&nonces);
        assert!(matches!(v, NonceVerdict::DuplicateAt { index: 2 }));
    }

    #[test]
    fn wrong_length_detected() {
        let nonces = vec![vec![0u8; 12], vec![0u8; 8]];
        let v = check(&nonces);
        assert!(matches!(v, NonceVerdict::WrongLength { index: 1, len: 8 }));
    }

    #[test]
    fn zero_length_nonce_rejected() {
        let nonces = vec![vec![]];
        assert!(matches!(
            check(&nonces),
            NonceVerdict::WrongLength { len: 0, .. }
        ));
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), NonceVerdict::Empty);
    }

    #[test]
    fn single_valid_nonce_passes() {
        assert_eq!(check(&[vec![0u8; 12]]), NonceVerdict::Ok);
    }

    #[test]
    fn duplicate_takes_priority_over_length_check() {
        // Length check happens first per nonce; valid → check uniqueness.
        // But if a later nonce has wrong length, we should report length first
        // (the inner loop checks length before insert).
        let nonces = vec![vec![0u8; 12], vec![0u8; 10]];
        assert!(matches!(
            check(&nonces),
            NonceVerdict::WrongLength { index: 1, .. }
        ));
    }

    #[test]
    fn many_unique_nonces_scale() {
        let many: Vec<Vec<u8>> = (0..1000u16)
            .map(|i| {
                let mut n = vec![0u8; 12];
                n[0..2].copy_from_slice(&i.to_le_bytes());
                n
            })
            .collect();
        assert_eq!(check(&many), NonceVerdict::Ok);
    }
}
