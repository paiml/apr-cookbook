//! # apr decrypt — AEAD Tag Verification
//!
//! `apr decrypt` uses AES-256-GCM, an AEAD scheme: every ciphertext carries
//! a 16-byte authentication tag. Decryption MUST verify the tag before
//! emitting any plaintext bytes — failing to do so would let a tampered
//! file silently produce corrupted weights. This recipe models the verify-
//! before-emit decision with a pure function so the contract can be
//! exercised in tests without doing real crypto.
//!
//! Demonstrates the **DECRYPT.5** recipe for PMAT-095 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST SP 800-38D §6.4 (Authenticated Decryption)
//!
//! Run with: cargo run --example cli_decrypt_aead_tag_verification
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct AeadFrame<'a> {
    pub nonce: &'a [u8; 12],
    pub ciphertext: &'a [u8],
    pub tag: &'a [u8; 16],
    pub aad: &'a [u8], // associated data — NOT encrypted but covered by tag
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecryptVerdict {
    Authenticated { plaintext_len: usize },
    TagMismatch,
    NonceTooShort,
    EmptyCiphertext,
}

/// Simulated AEAD decrypt — performs the verify-then-emit policy without
/// actually doing crypto. The simulated tag is `blake3_first_16(key || nonce || ciphertext || aad)`.
pub fn decrypt_aead(key: &[u8; 32], frame: &AeadFrame<'_>) -> DecryptVerdict {
    if frame.ciphertext.is_empty() {
        return DecryptVerdict::EmptyCiphertext;
    }
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(frame.nonce);
    h.update(frame.ciphertext);
    h.update(frame.aad);
    let computed = h.finalize();
    if &computed.as_bytes()[..16] != frame.tag {
        return DecryptVerdict::TagMismatch;
    }
    DecryptVerdict::Authenticated {
        plaintext_len: frame.ciphertext.len(),
    }
}

fn correct_tag(key: &[u8; 32], nonce: &[u8; 12], ct: &[u8], aad: &[u8]) -> [u8; 16] {
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(nonce);
    h.update(ct);
    h.update(aad);
    let mut tag = [0u8; 16];
    tag.copy_from_slice(&h.finalize().as_bytes()[..16]);
    tag
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_aead_tag_verification")?;

    let key = [0x42u8; 32];
    let nonce = [0x01u8; 12];
    let aad = b"file=model.apr";
    let ct: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
    let tag = correct_tag(&key, &nonce, &ct, aad);

    let happy = AeadFrame {
        nonce: &nonce,
        ciphertext: &ct,
        tag: &tag,
        aad,
    };
    let mut bad_tag = tag;
    bad_tag[0] ^= 0x01;
    let tampered = AeadFrame {
        nonce: &nonce,
        ciphertext: &ct,
        tag: &bad_tag,
        aad,
    };
    let empty = AeadFrame {
        nonce: &nonce,
        ciphertext: &[],
        tag: &tag,
        aad,
    };

    println!("happy:    {:?}", decrypt_aead(&key, &happy));
    println!("tampered: {:?}", decrypt_aead(&key, &tampered));
    println!("empty:    {:?}", decrypt_aead(&key, &empty));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture<'a>(
        nonce: &'a [u8; 12],
        ct: &'a [u8],
        tag: &'a [u8; 16],
        aad: &'a [u8],
    ) -> AeadFrame<'a> {
        AeadFrame {
            nonce,
            ciphertext: ct,
            tag,
            aad,
        }
    }

    #[test]
    fn aead_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn correct_tag_authenticates() {
        let key = [0xabu8; 32];
        let nonce = [0u8; 12];
        let ct = b"ciphertext";
        let aad = b"meta";
        let tag = correct_tag(&key, &nonce, ct, aad);
        let frame = fixture(&nonce, ct, &tag, aad);
        assert_eq!(
            decrypt_aead(&key, &frame),
            DecryptVerdict::Authenticated {
                plaintext_len: ct.len()
            }
        );
    }

    #[test]
    fn tampered_tag_rejected() {
        let key = [0xabu8; 32];
        let nonce = [0u8; 12];
        let ct = b"ciphertext";
        let aad = b"meta";
        let mut tag = correct_tag(&key, &nonce, ct, aad);
        tag[0] ^= 0xff;
        let frame = fixture(&nonce, ct, &tag, aad);
        assert_eq!(decrypt_aead(&key, &frame), DecryptVerdict::TagMismatch);
    }

    #[test]
    fn modified_aad_rejected() {
        // AAD is covered by the tag — changing it post-encryption invalidates.
        let key = [0xabu8; 32];
        let nonce = [0u8; 12];
        let ct = b"ciphertext";
        let tag = correct_tag(&key, &nonce, ct, b"original");
        let frame = fixture(&nonce, ct, &tag, b"modified");
        assert_eq!(decrypt_aead(&key, &frame), DecryptVerdict::TagMismatch);
    }

    #[test]
    fn modified_ciphertext_rejected() {
        // Same nonce + tag but flipped ciphertext byte — must fail verification.
        let key = [0xabu8; 32];
        let nonce = [0u8; 12];
        let original = b"ciphertext";
        let tag = correct_tag(&key, &nonce, original, b"meta");
        let mut tampered = original.to_vec();
        tampered[0] ^= 0xff;
        let frame = fixture(&nonce, &tampered, &tag, b"meta");
        assert_eq!(decrypt_aead(&key, &frame), DecryptVerdict::TagMismatch);
    }

    #[test]
    fn empty_ciphertext_short_circuits() {
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let tag = [0u8; 16];
        let frame = fixture(&nonce, &[], &tag, b"");
        assert_eq!(decrypt_aead(&key, &frame), DecryptVerdict::EmptyCiphertext);
    }
}
