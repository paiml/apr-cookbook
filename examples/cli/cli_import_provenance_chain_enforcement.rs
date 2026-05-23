//! # apr import — `--enforce-provenance` Chain Check
//!
//! `apr import <SOURCE> --enforce-provenance` rejects pre-baked GGUF
//! imports per F-GT-001 (single-provenance testing). Only SafeTensors
//! sources are allowed because their tensor bytes can be traced back
//! through `apr stamp` to a verifiable HF source. This recipe builds the
//! source-chain classifier and enforces the rule.
//!
//! Demonstrates the **IMPORT.5** recipe for PMAT-099 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender F-GT-001 + APR-PUB-001 provenance chain
//!
//! Run with: cargo run --example cli_import_provenance_chain_enforcement
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    SafeTensors,
    Gguf,
    Pytorch,
    Onnx,
}

#[derive(Debug, PartialEq)]
pub enum ProvenanceVerdict {
    Allowed,
    RejectedNonGold {
        format: SourceFormat,
        reason: &'static str,
    },
}

pub fn check_provenance(format: SourceFormat, enforce: bool) -> ProvenanceVerdict {
    if !enforce {
        return ProvenanceVerdict::Allowed;
    }
    match format {
        SourceFormat::SafeTensors => ProvenanceVerdict::Allowed,
        SourceFormat::Gguf => ProvenanceVerdict::RejectedNonGold {
            format,
            reason: "pre-baked GGUF can't be traced to original HF tensor bytes",
        },
        SourceFormat::Pytorch => ProvenanceVerdict::RejectedNonGold {
            format,
            reason: "pickled tensors lack a stamped lineage record",
        },
        SourceFormat::Onnx => ProvenanceVerdict::RejectedNonGold {
            format,
            reason: "ONNX graph rewrites lose original layer identities",
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_provenance_chain_enforcement")?;

    for fmt in [
        SourceFormat::SafeTensors,
        SourceFormat::Gguf,
        SourceFormat::Pytorch,
        SourceFormat::Onnx,
    ] {
        for enforce in [false, true] {
            println!(
                "{fmt:>15?}  enforce={enforce:>5}  →  {:?}",
                check_provenance(fmt, enforce)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn enforce_off_always_allows() {
        for fmt in [
            SourceFormat::SafeTensors,
            SourceFormat::Gguf,
            SourceFormat::Pytorch,
            SourceFormat::Onnx,
        ] {
            assert_eq!(check_provenance(fmt, false), ProvenanceVerdict::Allowed);
        }
    }

    #[test]
    fn safetensors_always_allowed_under_enforcement() {
        // SafeTensors is the only gold-standard provenance source.
        assert_eq!(
            check_provenance(SourceFormat::SafeTensors, true),
            ProvenanceVerdict::Allowed
        );
    }

    #[test]
    fn gguf_rejected_under_enforcement() {
        let v = check_provenance(SourceFormat::Gguf, true);
        assert!(matches!(v, ProvenanceVerdict::RejectedNonGold { .. }));
    }

    #[test]
    fn pytorch_rejected_under_enforcement() {
        let v = check_provenance(SourceFormat::Pytorch, true);
        assert!(matches!(v, ProvenanceVerdict::RejectedNonGold { .. }));
    }

    #[test]
    fn rejection_reason_is_format_specific() {
        // Each rejected format must have a distinct reason — generic "rejected"
        // wouldn't help the operator pick the right replacement.
        let g = check_provenance(SourceFormat::Gguf, true);
        let p = check_provenance(SourceFormat::Pytorch, true);
        let o = check_provenance(SourceFormat::Onnx, true);
        let reasons: Vec<&str> = [g, p, o]
            .iter()
            .filter_map(|v| match v {
                ProvenanceVerdict::RejectedNonGold { reason, .. } => Some(*reason),
                _ => None,
            })
            .collect();
        assert_eq!(reasons.len(), 3);
        let unique: std::collections::HashSet<&str> = reasons.iter().copied().collect();
        assert_eq!(unique.len(), 3, "reasons must be distinct: {reasons:?}");
    }
}
