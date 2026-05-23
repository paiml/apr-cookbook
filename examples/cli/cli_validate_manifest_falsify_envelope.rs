//! # apr validate-manifest — FALSIFY-PM-001..006 Envelope
//!
//! `apr validate-manifest` discharges six FALSIFY-PM gates: schema
//! shape (PM-001), local sha256 match (PM-002), live URL fetch
//! (PM-003), parent-chain integrity (PM-004), provenance fields (PM-005),
//! signature (PM-006). This recipe builds the discharge accounting
//! and asserts the contract.
//!
//! Demonstrates the **VAL-MANIFEST.6** recipe for PMAT-110 (apr validate-manifest coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts/publish-manifest-v1.yaml + SPEC-SHIP-TWO-001 §12.3
//!
//! Run with: cargo run --example cli_validate_manifest_falsify_envelope
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gate {
    Pm001Schema,
    Pm002LocalSha256,
    Pm003LiveFetch,
    Pm004ParentChain,
    Pm005Provenance,
    Pm006Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Discharged,
    Deferred,
    Failed,
    NotApplicable,
}

#[derive(Debug, Default, Clone)]
pub struct DischargeFlags {
    pub artifact_provided: bool,
    pub live_mode: bool,
    pub offline: bool,
    pub has_signature: bool,
}

pub fn evaluate_gate(gate: Gate, flags: &DischargeFlags) -> GateState {
    match gate {
        // Schema: always evaluated.
        Gate::Pm001Schema => GateState::Discharged,
        // Local sha256: needs --artifact.
        Gate::Pm002LocalSha256 => {
            if flags.artifact_provided {
                GateState::Discharged
            } else {
                GateState::Deferred
            }
        }
        // Live: needs --live AND not --offline.
        Gate::Pm003LiveFetch => {
            if flags.offline {
                GateState::NotApplicable
            } else if flags.live_mode {
                GateState::Discharged
            } else {
                GateState::Deferred
            }
        }
        // Parent chain: schema-only, always evaluated.
        Gate::Pm004ParentChain => GateState::Discharged,
        // Provenance: schema-only.
        Gate::Pm005Provenance => GateState::Discharged,
        // Signature: needs sig present in manifest.
        Gate::Pm006Signature => {
            if flags.has_signature {
                GateState::Discharged
            } else {
                GateState::Deferred
            }
        }
    }
}

pub fn full_envelope(flags: &DischargeFlags) -> Vec<(Gate, GateState)> {
    [
        Gate::Pm001Schema,
        Gate::Pm002LocalSha256,
        Gate::Pm003LiveFetch,
        Gate::Pm004ParentChain,
        Gate::Pm005Provenance,
        Gate::Pm006Signature,
    ]
    .iter()
    .map(|g| (*g, evaluate_gate(*g, flags)))
    .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_manifest_falsify_envelope")?;

    let cases = [
        (
            "default (offline)",
            DischargeFlags {
                artifact_provided: false,
                live_mode: false,
                offline: false,
                has_signature: false,
            },
        ),
        (
            "with artifact",
            DischargeFlags {
                artifact_provided: true,
                live_mode: false,
                offline: false,
                has_signature: false,
            },
        ),
        (
            "live mode",
            DischargeFlags {
                artifact_provided: true,
                live_mode: true,
                offline: false,
                has_signature: true,
            },
        ),
        (
            "offline overrides live",
            DischargeFlags {
                artifact_provided: true,
                live_mode: true,
                offline: true,
                has_signature: true,
            },
        ),
    ];

    for (label, f) in cases {
        println!("{label}:");
        for (gate, state) in full_envelope(&f) {
            println!("  {gate:?}: {state:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn schema_always_discharged() {
        let f = DischargeFlags::default();
        assert_eq!(evaluate_gate(Gate::Pm001Schema, &f), GateState::Discharged);
    }

    #[test]
    fn local_sha_needs_artifact() {
        let no_art = DischargeFlags::default();
        assert_eq!(
            evaluate_gate(Gate::Pm002LocalSha256, &no_art),
            GateState::Deferred
        );
        let with_art = DischargeFlags {
            artifact_provided: true,
            ..Default::default()
        };
        assert_eq!(
            evaluate_gate(Gate::Pm002LocalSha256, &with_art),
            GateState::Discharged
        );
    }

    #[test]
    fn live_requires_not_offline() {
        let live_off = DischargeFlags {
            live_mode: true,
            offline: true,
            ..Default::default()
        };
        // --offline overrides --live → NotApplicable.
        assert_eq!(
            evaluate_gate(Gate::Pm003LiveFetch, &live_off),
            GateState::NotApplicable
        );
    }

    #[test]
    fn signature_needs_signature_in_manifest() {
        let no_sig = DischargeFlags::default();
        assert_eq!(
            evaluate_gate(Gate::Pm006Signature, &no_sig),
            GateState::Deferred
        );
        let with_sig = DischargeFlags {
            has_signature: true,
            ..Default::default()
        };
        assert_eq!(
            evaluate_gate(Gate::Pm006Signature, &with_sig),
            GateState::Discharged
        );
    }

    #[test]
    fn full_envelope_returns_six_gates() {
        let f = DischargeFlags::default();
        let env = full_envelope(&f);
        assert_eq!(env.len(), 6);
    }

    #[test]
    fn parent_and_provenance_always_discharged() {
        let f = DischargeFlags::default();
        assert_eq!(
            evaluate_gate(Gate::Pm004ParentChain, &f),
            GateState::Discharged
        );
        assert_eq!(
            evaluate_gate(Gate::Pm005Provenance, &f),
            GateState::Discharged
        );
    }
}
