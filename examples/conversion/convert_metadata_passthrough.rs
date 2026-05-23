//! # Conversion Metadata Passthrough
//!
//! Free-form metadata preserved across format conversion. Rules:
//!   well-known keys (architecture, vocab_size) → required-pass
//!   numeric stats (mean, std) → derive if missing
//!   user keys (custom.*) → pass-through opaque
//!
//! Picker classifies each key + reports lost keys (when target format
//! doesn't support them).
//!
//! Demonstrates the **CONV.17** recipe for PMAT-148 (conversion round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace tokenizer config + GGUF metadata conventions.
//!
//! Run with: cargo run --example convert_metadata_passthrough
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFormat {
    Apr2,
    Gguf,
    SafeTensors,
    Onnx,
}

#[derive(Debug, PartialEq)]
pub enum PassthroughVerdict {
    Ok {
        kept_keys: Vec<String>,
        lost_keys: Vec<String>,
        derived_keys: Vec<String>,
    },
    EmptyMetadata,
}

const REQUIRED_KEYS: &[&str] = &["architecture", "vocab_size", "context_length"];

pub fn classify(metadata_keys: &[&str], target: TargetFormat) -> PassthroughVerdict {
    if metadata_keys.is_empty() {
        return PassthroughVerdict::EmptyMetadata;
    }
    let mut kept = Vec::new();
    let mut lost = Vec::new();
    let mut derived = Vec::new();
    for &k in metadata_keys {
        if REQUIRED_KEYS.contains(&k) {
            kept.push(k.to_string());
        } else if k.starts_with("custom.") {
            // Only APR2/SafeTensors preserve free-form custom keys.
            if matches!(target, TargetFormat::Apr2 | TargetFormat::SafeTensors) {
                kept.push(k.to_string());
            } else {
                lost.push(k.to_string());
            }
        } else if k == "mean" || k == "std" {
            kept.push(k.to_string());
        } else if matches!(target, TargetFormat::Onnx) && !is_onnx_compatible(k) {
            lost.push(k.to_string());
        } else {
            kept.push(k.to_string());
        }
    }
    // Derive missing required keys.
    for &req in REQUIRED_KEYS {
        if !metadata_keys.contains(&req) {
            derived.push(req.to_string());
        }
    }
    PassthroughVerdict::Ok {
        kept_keys: kept,
        lost_keys: lost,
        derived_keys: derived,
    }
}

fn is_onnx_compatible(key: &str) -> bool {
    matches!(
        key,
        "architecture" | "vocab_size" | "context_length" | "mean" | "std"
    )
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_metadata_passthrough")?;

    let keys = [
        "architecture",
        "vocab_size",
        "custom.notes",
        "training_date",
    ];
    println!("APR2: {:?}", classify(&keys, TargetFormat::Apr2));
    println!("GGUF: {:?}", classify(&keys, TargetFormat::Gguf));
    println!("ONNX: {:?}", classify(&keys, TargetFormat::Onnx));
    println!("empty: {:?}", classify(&[], TargetFormat::Apr2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_metadata_rejected() {
        assert_eq!(
            classify(&[], TargetFormat::Apr2),
            PassthroughVerdict::EmptyMetadata
        );
    }

    #[test]
    fn required_keys_kept() {
        let v = classify(&["architecture", "vocab_size"], TargetFormat::Gguf);
        if let PassthroughVerdict::Ok { kept_keys, .. } = v {
            assert!(kept_keys.contains(&"architecture".to_string()));
            assert!(kept_keys.contains(&"vocab_size".to_string()));
        }
    }

    #[test]
    fn custom_keys_pass_through_apr2() {
        let v = classify(&["custom.foo"], TargetFormat::Apr2);
        if let PassthroughVerdict::Ok { kept_keys, .. } = v {
            assert!(kept_keys.contains(&"custom.foo".to_string()));
        }
    }

    #[test]
    fn custom_keys_lost_in_gguf() {
        let v = classify(&["architecture", "custom.foo"], TargetFormat::Gguf);
        if let PassthroughVerdict::Ok { lost_keys, .. } = v {
            assert!(lost_keys.contains(&"custom.foo".to_string()));
        }
    }

    #[test]
    fn missing_required_derived() {
        let v = classify(&["random_key"], TargetFormat::Apr2);
        if let PassthroughVerdict::Ok { derived_keys, .. } = v {
            // All REQUIRED_KEYS missing.
            assert_eq!(derived_keys.len(), REQUIRED_KEYS.len());
        }
    }

    #[test]
    fn no_derived_when_all_required_present() {
        let v = classify(&REQUIRED_KEYS, TargetFormat::Apr2);
        if let PassthroughVerdict::Ok { derived_keys, .. } = v {
            assert!(derived_keys.is_empty());
        }
    }

    #[test]
    fn safetensors_keeps_custom() {
        let v = classify(&["custom.bar"], TargetFormat::SafeTensors);
        if let PassthroughVerdict::Ok { kept_keys, .. } = v {
            assert!(kept_keys.contains(&"custom.bar".to_string()));
        }
    }

    #[test]
    fn onnx_loses_unsupported() {
        let v = classify(&["random_user_key"], TargetFormat::Onnx);
        if let PassthroughVerdict::Ok { lost_keys, .. } = v {
            assert!(lost_keys.contains(&"random_user_key".to_string()));
        }
    }

    #[test]
    fn mean_std_kept_in_all_formats() {
        for tgt in [
            TargetFormat::Apr2,
            TargetFormat::Gguf,
            TargetFormat::SafeTensors,
            TargetFormat::Onnx,
        ] {
            let v = classify(&["mean", "std"], tgt);
            if let PassthroughVerdict::Ok { kept_keys, .. } = v {
                assert!(kept_keys.contains(&"mean".to_string()));
                assert!(kept_keys.contains(&"std".to_string()));
            }
        }
    }

    #[test]
    fn arch_kept_across_formats() {
        for tgt in [
            TargetFormat::Apr2,
            TargetFormat::Gguf,
            TargetFormat::SafeTensors,
            TargetFormat::Onnx,
        ] {
            let v = classify(&["architecture"], tgt);
            if let PassthroughVerdict::Ok { kept_keys, .. } = v {
                assert!(kept_keys.contains(&"architecture".to_string()));
            }
        }
    }
}
