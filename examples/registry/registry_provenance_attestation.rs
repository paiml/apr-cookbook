//! # Registry SLSA Provenance Attestation Verifier
//!
//! SLSA (Supply-chain Levels for Software Artifacts):
//!   L1: documented build process
//!   L2: hosted build + signed provenance
//!   L3: hardened build + immutable history
//!   L4: hermetic build + reproducible
//!
//! Verifier checks claimed level against attestation fields and emits
//! the highest verifiable level.
//!
//! Demonstrates the **REG.19** recipe for PMAT-147 (registry round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA Framework v1.0 specification.
//!
//! Run with: cargo run --example registry_provenance_attestation
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SlsaLevel {
    L1,
    L2,
    L3,
    L4,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AttestationFields {
    pub has_build_process_doc: bool,
    pub has_signed_provenance: bool,
    pub uses_hosted_builder: bool,
    pub has_immutable_history: bool,
    pub is_hermetic: bool,
    pub is_reproducible: bool,
}

#[derive(Debug, PartialEq)]
pub enum AttestationVerdict {
    Ok {
        verified_level: SlsaLevel,
    },
    Unverified {
        missing: Vec<&'static str>,
    },
    LevelDowngrade {
        claimed: SlsaLevel,
        verifiable: SlsaLevel,
    },
}

pub fn verify(claimed: SlsaLevel, fields: AttestationFields) -> AttestationVerdict {
    let Some(verifiable) = compute_max(fields) else {
        return AttestationVerdict::Unverified {
            missing: vec!["build_process_doc"],
        };
    };
    if verifiable < claimed {
        return AttestationVerdict::LevelDowngrade {
            claimed,
            verifiable,
        };
    }
    AttestationVerdict::Ok {
        verified_level: verifiable,
    }
}

fn compute_max(fields: AttestationFields) -> Option<SlsaLevel> {
    if !fields.has_build_process_doc {
        return None;
    }
    if !(fields.has_signed_provenance && fields.uses_hosted_builder) {
        return Some(SlsaLevel::L1);
    }
    if !fields.has_immutable_history {
        return Some(SlsaLevel::L2);
    }
    if !(fields.is_hermetic && fields.is_reproducible) {
        return Some(SlsaLevel::L3);
    }
    Some(SlsaLevel::L4)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_provenance_attestation")?;

    let l1 = AttestationFields {
        has_build_process_doc: true,
        ..Default::default()
    };
    println!("L1 fields, claim L1: {:?}", verify(SlsaLevel::L1, l1));

    let l3 = AttestationFields {
        has_build_process_doc: true,
        has_signed_provenance: true,
        uses_hosted_builder: true,
        has_immutable_history: true,
        ..Default::default()
    };
    println!("L3 fields, claim L3: {:?}", verify(SlsaLevel::L3, l3));
    println!(
        "L3 fields, claim L4 (downgrade): {:?}",
        verify(SlsaLevel::L4, l3)
    );

    let nothing = AttestationFields::default();
    println!("Nothing: {:?}", verify(SlsaLevel::L1, nothing));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> AttestationFields {
        AttestationFields {
            has_build_process_doc: true,
            has_signed_provenance: true,
            uses_hosted_builder: true,
            has_immutable_history: true,
            is_hermetic: true,
            is_reproducible: true,
        }
    }

    #[test]
    fn verifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn l4_full_fields_verified() {
        let v = verify(SlsaLevel::L4, full());
        assert_eq!(
            v,
            AttestationVerdict::Ok {
                verified_level: SlsaLevel::L4
            }
        );
    }

    #[test]
    fn l1_basic_doc_verified() {
        let f = AttestationFields {
            has_build_process_doc: true,
            ..Default::default()
        };
        let v = verify(SlsaLevel::L1, f);
        assert!(matches!(v, AttestationVerdict::Ok { .. }));
    }

    #[test]
    fn nothing_unverified() {
        let v = verify(SlsaLevel::L1, AttestationFields::default());
        assert!(matches!(v, AttestationVerdict::Unverified { .. }));
    }

    #[test]
    fn claim_too_high_downgrade() {
        let f = AttestationFields {
            has_build_process_doc: true,
            ..Default::default()
        };
        let v = verify(SlsaLevel::L4, f);
        assert!(matches!(v, AttestationVerdict::LevelDowngrade { .. }));
    }

    #[test]
    fn claim_below_capability_ok() {
        let v = verify(SlsaLevel::L2, full());
        assert!(matches!(v, AttestationVerdict::Ok { .. }));
    }

    #[test]
    fn missing_signature_caps_at_l1() {
        let f = AttestationFields {
            has_build_process_doc: true,
            has_signed_provenance: false,
            uses_hosted_builder: true,
            ..Default::default()
        };
        let v = verify(SlsaLevel::L2, f);
        assert!(matches!(v, AttestationVerdict::LevelDowngrade { .. }));
    }

    #[test]
    fn missing_immutable_caps_at_l2() {
        let f = AttestationFields {
            has_build_process_doc: true,
            has_signed_provenance: true,
            uses_hosted_builder: true,
            ..Default::default()
        };
        let v = verify(SlsaLevel::L3, f);
        assert!(matches!(v, AttestationVerdict::LevelDowngrade { .. }));
    }

    #[test]
    fn missing_hermetic_caps_at_l3() {
        let f = AttestationFields {
            has_build_process_doc: true,
            has_signed_provenance: true,
            uses_hosted_builder: true,
            has_immutable_history: true,
            is_hermetic: false,
            is_reproducible: true,
        };
        let v = verify(SlsaLevel::L4, f);
        assert!(matches!(v, AttestationVerdict::LevelDowngrade { .. }));
    }

    #[test]
    fn levels_ordered() {
        assert!(SlsaLevel::L1 < SlsaLevel::L2);
        assert!(SlsaLevel::L2 < SlsaLevel::L3);
        assert!(SlsaLevel::L3 < SlsaLevel::L4);
    }

    #[test]
    fn l3_full_chain_verified() {
        let mut f = full();
        f.is_hermetic = false;
        f.is_reproducible = false;
        let v = verify(SlsaLevel::L3, f);
        assert_eq!(
            v,
            AttestationVerdict::Ok {
                verified_level: SlsaLevel::L3
            }
        );
    }
}
