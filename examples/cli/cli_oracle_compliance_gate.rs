//! # apr oracle — Compliance Gate
//!
//! `apr oracle --compliance <FILE>` runs a contract-compliance check that
//! exits non-zero if the model deviates from its declared family. This
//! recipe models the gate as a pure function: given a (declared family,
//! observed signature) pair, it returns Pass / Drift / FamilyMismatch.
//! This lets CI preview the verdict before invoking the binary.
//!
//! Demonstrates the **ORACLE.4** recipe for PMAT-093 (apr oracle coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ORACLE-002 + GATE-APR-CONTRACT-001
//!
//! Run with: cargo run --example cli_oracle_compliance_gate
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSignature {
    pub family: String,
    pub vocab: u32,
    pub n_layers: u32,
    pub hidden: u32,
    pub q_proj_path: String,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceVerdict {
    Pass,
    Drift {
        field: &'static str,
        expected: String,
        observed: String,
    },
    FamilyMismatch {
        declared: String,
        likely: String,
    },
}

pub fn check_compliance(declared: &str, sig: &ModelSignature) -> ComplianceVerdict {
    if declared != sig.family {
        return ComplianceVerdict::FamilyMismatch {
            declared: declared.into(),
            likely: sig.family.clone(),
        };
    }
    // Family-specific schema invariants.
    match declared {
        "qwen2" => {
            if !sig.q_proj_path.contains("model.layers") {
                return ComplianceVerdict::Drift {
                    field: "q_proj_path",
                    expected: "model.layers.{n}.self_attn.q_proj.weight".into(),
                    observed: sig.q_proj_path.clone(),
                };
            }
            if sig.dtype != "bf16" && sig.dtype != "fp16" {
                return ComplianceVerdict::Drift {
                    field: "dtype",
                    expected: "bf16 or fp16".into(),
                    observed: sig.dtype.clone(),
                };
            }
        }
        "whisper" if !sig.q_proj_path.contains("encoder.layers") => {
            return ComplianceVerdict::Drift {
                field: "q_proj_path",
                expected: "model.encoder.layers.{n}.self_attn.q_proj.weight".into(),
                observed: sig.q_proj_path.clone(),
            };
        }
        _ => {}
    }
    ComplianceVerdict::Pass
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_oracle_compliance_gate")?;

    let happy = ModelSignature {
        family: "qwen2".into(),
        vocab: 152_064,
        n_layers: 28,
        hidden: 3584,
        q_proj_path: "model.layers.0.self_attn.q_proj.weight".into(),
        dtype: "bf16".into(),
    };
    let drifted = ModelSignature {
        family: "qwen2".into(),
        vocab: 152_064,
        n_layers: 28,
        hidden: 3584,
        q_proj_path: "encoder.layer.0.attention.self.query.weight".into(), // BERT-style ⚠
        dtype: "bf16".into(),
    };
    let mismatched = ModelSignature {
        family: "llama".into(), // signature says LLaMA but operator declared qwen2
        vocab: 128_256,
        n_layers: 32,
        hidden: 4096,
        q_proj_path: "model.layers.0.self_attn.q_proj.weight".into(),
        dtype: "bf16".into(),
    };

    println!("happy:       {:?}", check_compliance("qwen2", &happy));
    println!("drifted:     {:?}", check_compliance("qwen2", &drifted));
    println!("mismatched:  {:?}", check_compliance("qwen2", &mismatched));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_sig() -> ModelSignature {
        ModelSignature {
            family: "qwen2".into(),
            vocab: 152_064,
            n_layers: 28,
            hidden: 3584,
            q_proj_path: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: "bf16".into(),
        }
    }

    #[test]
    fn compliance_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn matched_signature_passes() {
        assert_eq!(
            check_compliance("qwen2", &qwen_sig()),
            ComplianceVerdict::Pass
        );
    }

    #[test]
    fn family_mismatch_reported_first() {
        // Even if other fields look fine, family mismatch wins (highest priority).
        let mut sig = qwen_sig();
        sig.family = "llama".into();
        let v = check_compliance("qwen2", &sig);
        assert!(matches!(v, ComplianceVerdict::FamilyMismatch { .. }));
    }

    #[test]
    fn schema_drift_detected_when_family_matches() {
        let mut sig = qwen_sig();
        sig.q_proj_path = "blocks.0.attn.qkv.weight".into(); // GPT-style
        let v = check_compliance("qwen2", &sig);
        assert!(matches!(
            v,
            ComplianceVerdict::Drift {
                field: "q_proj_path",
                ..
            }
        ));
    }

    #[test]
    fn whisper_demands_encoder_path() {
        let sig = ModelSignature {
            family: "whisper".into(),
            vocab: 51_865,
            n_layers: 32,
            hidden: 1280,
            q_proj_path: "model.layers.0.self_attn.q_proj.weight".into(), // wrong (LLaMA-style)
            dtype: "fp16".into(),
        };
        let v = check_compliance("whisper", &sig);
        assert!(matches!(
            v,
            ComplianceVerdict::Drift {
                field: "q_proj_path",
                ..
            }
        ));
    }

    #[test]
    fn unknown_family_passes_vacuously() {
        // Oracle has no schema spec for unknown families — pass-through, with
        // a warning emitted elsewhere (out of scope for this rule).
        let sig = ModelSignature {
            family: "exotic".into(),
            vocab: 1024,
            n_layers: 1,
            hidden: 64,
            q_proj_path: "anywhere.qkv".into(),
            dtype: "fp32".into(),
        };
        assert_eq!(check_compliance("exotic", &sig), ComplianceVerdict::Pass);
    }
}
