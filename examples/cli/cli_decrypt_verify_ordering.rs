//! # apr decrypt — Verify-Then-Decrypt Ordering Enforcer
//!
//! AEAD ciphers (AES-GCM, ChaCha20-Poly1305) MUST verify the
//! authentication tag BEFORE returning plaintext. Returning
//! plaintext-then-verifying enables chosen-ciphertext attacks. This
//! recipe codifies the ordering as a state machine.
//!
//! Demonstrates the **DEC.5** recipe for PMAT-121 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEC-001 + Bellare & Namprempre 2000 (encrypt-then-MAC)
//!
//! Run with: cargo run --example cli_decrypt_verify_ordering
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecryptStage {
    Initial,
    HeaderRead,
    TagVerified,
    PlaintextEmitted,
    Failed,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok(DecryptStage),
    InvalidTransition {
        from: DecryptStage,
        to: DecryptStage,
    },
    InsecureOrdering, // emitting plaintext before tag verification
}

pub fn transition(from: DecryptStage, to: DecryptStage) -> TransitionVerdict {
    let valid = matches!(
        (from, to),
        (DecryptStage::Initial, DecryptStage::HeaderRead)
            | (
                DecryptStage::HeaderRead,
                DecryptStage::TagVerified | DecryptStage::Failed
            )
            | (
                DecryptStage::TagVerified,
                DecryptStage::PlaintextEmitted | DecryptStage::Failed,
            )
    );
    if valid {
        TransitionVerdict::Ok(to)
    } else if from == DecryptStage::HeaderRead && to == DecryptStage::PlaintextEmitted {
        TransitionVerdict::InsecureOrdering
    } else {
        TransitionVerdict::InvalidTransition { from, to }
    }
}

pub fn run_sequence(stages: &[DecryptStage]) -> Vec<TransitionVerdict> {
    let mut out = Vec::new();
    for w in stages.windows(2) {
        out.push(transition(w[0], w[1]));
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_verify_ordering")?;

    let secure = [
        DecryptStage::Initial,
        DecryptStage::HeaderRead,
        DecryptStage::TagVerified,
        DecryptStage::PlaintextEmitted,
    ];
    println!("secure: {:?}", run_sequence(&secure));

    let insecure = [
        DecryptStage::Initial,
        DecryptStage::HeaderRead,
        DecryptStage::PlaintextEmitted, // skipped TagVerified — INSECURE
    ];
    println!("insecure: {:?}", run_sequence(&insecure));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforcer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn secure_full_sequence_passes() {
        let seq = [
            DecryptStage::Initial,
            DecryptStage::HeaderRead,
            DecryptStage::TagVerified,
            DecryptStage::PlaintextEmitted,
        ];
        let r = run_sequence(&seq);
        assert!(r.iter().all(|t| matches!(t, TransitionVerdict::Ok(_))));
    }

    #[test]
    fn skipping_tag_verification_flagged() {
        let seq = [
            DecryptStage::Initial,
            DecryptStage::HeaderRead,
            DecryptStage::PlaintextEmitted,
        ];
        let r = run_sequence(&seq);
        assert!(matches!(r[1], TransitionVerdict::InsecureOrdering));
    }

    #[test]
    fn header_to_failed_allowed() {
        // Failure path before tag verification is acceptable (e.g., bad header).
        let v = transition(DecryptStage::HeaderRead, DecryptStage::Failed);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn tag_verified_to_failed_allowed() {
        // Failure during plaintext write is also acceptable.
        let v = transition(DecryptStage::TagVerified, DecryptStage::Failed);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn initial_to_tag_verified_invalid() {
        // Skipping HeaderRead → invalid transition.
        let v = transition(DecryptStage::Initial, DecryptStage::TagVerified);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn initial_to_plaintext_invalid() {
        let v = transition(DecryptStage::Initial, DecryptStage::PlaintextEmitted);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn cannot_re_enter_initial() {
        let v = transition(DecryptStage::HeaderRead, DecryptStage::Initial);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn empty_or_single_stage_yields_no_transitions() {
        assert!(run_sequence(&[]).is_empty());
        assert!(run_sequence(&[DecryptStage::Initial]).is_empty());
    }
}
