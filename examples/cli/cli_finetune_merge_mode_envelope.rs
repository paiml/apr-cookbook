//! # apr finetune --merge — Adapter Merge Envelope
//!
//! `apr finetune --merge --adapter <DIR> -o <OUT> <BASE>` merges a LoRA
//! adapter into the base model. The merge cannot be reversed, so the
//! envelope must validate: adapter dir exists, base model exists, output
//! ≠ base (would silently overwrite the base), output ≠ adapter (would
//! silently corrupt the adapter).
//!
//! Demonstrates the **FINETUNE.6** recipe for PMAT-104 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-244 + LoRA merge irreversibility
//!
//! Run with: cargo run --example cli_finetune_merge_mode_envelope
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone)]
pub struct MergeInvocation {
    pub base_path: String,
    pub adapter_path: String,
    pub output_path: String,
    pub merge_flag: bool,
}

#[derive(Debug, PartialEq)]
pub enum MergeVerdict {
    Ok,
    MergeFlagNotSet,
    EmptyPath,
    OutputSameAsBase,
    OutputSameAsAdapter,
    AdapterSameAsBase,
}

pub fn validate_merge(inv: &MergeInvocation) -> MergeVerdict {
    if !inv.merge_flag {
        return MergeVerdict::MergeFlagNotSet;
    }
    if inv.base_path.is_empty() || inv.adapter_path.is_empty() || inv.output_path.is_empty() {
        return MergeVerdict::EmptyPath;
    }
    if inv.output_path == inv.base_path {
        return MergeVerdict::OutputSameAsBase;
    }
    if inv.output_path == inv.adapter_path {
        return MergeVerdict::OutputSameAsAdapter;
    }
    if inv.adapter_path == inv.base_path {
        return MergeVerdict::AdapterSameAsBase;
    }
    MergeVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_merge_mode_envelope")?;

    let cases = [
        (
            "happy",
            MergeInvocation {
                base_path: "base.apr".into(),
                adapter_path: "adapter/".into(),
                output_path: "merged.apr".into(),
                merge_flag: true,
            },
        ),
        (
            "no merge flag",
            MergeInvocation {
                base_path: "base.apr".into(),
                adapter_path: "adapter/".into(),
                output_path: "merged.apr".into(),
                merge_flag: false,
            },
        ),
        (
            "out=base",
            MergeInvocation {
                base_path: "model.apr".into(),
                adapter_path: "adapter/".into(),
                output_path: "model.apr".into(),
                merge_flag: true,
            },
        ),
        (
            "out=adapter",
            MergeInvocation {
                base_path: "base.apr".into(),
                adapter_path: "adapter/".into(),
                output_path: "adapter/".into(),
                merge_flag: true,
            },
        ),
    ];

    for (label, inv) in cases {
        println!("{label:>15}  →  {:?}", validate_merge(&inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy() -> MergeInvocation {
        MergeInvocation {
            base_path: "base.apr".into(),
            adapter_path: "adapter/".into(),
            output_path: "merged.apr".into(),
            merge_flag: true,
        }
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_merge_passes() {
        assert_eq!(validate_merge(&happy()), MergeVerdict::Ok);
    }

    #[test]
    fn missing_merge_flag_rejected() {
        let mut inv = happy();
        inv.merge_flag = false;
        assert_eq!(validate_merge(&inv), MergeVerdict::MergeFlagNotSet);
    }

    #[test]
    fn empty_base_path_rejected() {
        let mut inv = happy();
        inv.base_path = String::new();
        assert_eq!(validate_merge(&inv), MergeVerdict::EmptyPath);
    }

    #[test]
    fn output_equal_to_base_rejected() {
        // Critical: must not silently overwrite the base model.
        let mut inv = happy();
        inv.output_path = inv.base_path.clone();
        assert_eq!(validate_merge(&inv), MergeVerdict::OutputSameAsBase);
    }

    #[test]
    fn output_equal_to_adapter_rejected() {
        let mut inv = happy();
        inv.output_path = inv.adapter_path.clone();
        assert_eq!(validate_merge(&inv), MergeVerdict::OutputSameAsAdapter);
    }

    #[test]
    fn adapter_equal_to_base_rejected() {
        // Pathological — operator confused which file is which.
        let mut inv = happy();
        inv.adapter_path = inv.base_path.clone();
        assert_eq!(validate_merge(&inv), MergeVerdict::AdapterSameAsBase);
    }
}
