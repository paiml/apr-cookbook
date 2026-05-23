//! # Conversion Tensor-Name Remapper
//!
//! Different formats use different naming conventions: HF
//! `model.layers.0.self_attn.q_proj.weight`; GGUF `blk.0.attn_q.weight`;
//! ONNX `Constant_1234`. Conversion needs a per-pair remapper. This
//! recipe builds the table-driven remapper + missing-name detector.
//!
//! Demonstrates the **CONV.8** recipe for PMAT-127 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GGUF naming convention; HuggingFace transformers naming.
//!
//! Run with: cargo run --example convert_tensor_name_remapper
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamingConvention {
    HuggingFace,
    Gguf,
    Onnx,
}

#[derive(Debug, PartialEq)]
pub enum RemapVerdict {
    Ok { mapped_name: String },
    UnknownPattern,
    InvalidSourceName,
}

pub fn remap(name: &str, from: NamingConvention, to: NamingConvention) -> RemapVerdict {
    if name.is_empty() {
        return RemapVerdict::InvalidSourceName;
    }
    if from == to {
        return RemapVerdict::Ok {
            mapped_name: name.to_string(),
        };
    }
    let parsed = match from {
        NamingConvention::HuggingFace => parse_hf(name),
        NamingConvention::Gguf => parse_gguf(name),
        NamingConvention::Onnx => return RemapVerdict::UnknownPattern,
    };
    let Some((layer_idx, role, suffix)) = parsed else {
        return RemapVerdict::UnknownPattern;
    };
    let mapped = match to {
        NamingConvention::HuggingFace => {
            let role_str = match role.as_str() {
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_o" => "self_attn.o_proj",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                _ => return RemapVerdict::UnknownPattern,
            };
            format!("model.layers.{layer_idx}.{role_str}.{suffix}")
        }
        NamingConvention::Gguf => {
            let role_str = match role.as_str() {
                "self_attn.q_proj" => "attn_q",
                "self_attn.k_proj" => "attn_k",
                "self_attn.v_proj" => "attn_v",
                "self_attn.o_proj" => "attn_o",
                "mlp.gate_proj" => "ffn_gate",
                "mlp.up_proj" => "ffn_up",
                "mlp.down_proj" => "ffn_down",
                _ => return RemapVerdict::UnknownPattern,
            };
            format!("blk.{layer_idx}.{role_str}.{suffix}")
        }
        NamingConvention::Onnx => return RemapVerdict::UnknownPattern,
    };
    RemapVerdict::Ok {
        mapped_name: mapped,
    }
}

fn parse_hf(name: &str) -> Option<(u32, String, String)> {
    // model.layers.{N}.{role}.{suffix}
    let rest = name.strip_prefix("model.layers.")?;
    let (idx_str, rest) = rest.split_once('.')?;
    let idx: u32 = idx_str.parse().ok()?;
    let (role, suffix) = rest.rsplit_once('.')?;
    Some((idx, role.to_string(), suffix.to_string()))
}

fn parse_gguf(name: &str) -> Option<(u32, String, String)> {
    // blk.{N}.{role}.{suffix}
    let rest = name.strip_prefix("blk.")?;
    let (idx_str, rest) = rest.split_once('.')?;
    let idx: u32 = idx_str.parse().ok()?;
    let (role, suffix) = rest.rsplit_once('.')?;
    Some((idx, role.to_string(), suffix.to_string()))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_tensor_name_remapper")?;

    let hf_to_gguf = remap(
        "model.layers.5.self_attn.q_proj.weight",
        NamingConvention::HuggingFace,
        NamingConvention::Gguf,
    );
    println!("HF → GGUF: {hf_to_gguf:?}");
    let gguf_to_hf = remap(
        "blk.5.attn_q.weight",
        NamingConvention::Gguf,
        NamingConvention::HuggingFace,
    );
    println!("GGUF → HF: {gguf_to_hf:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remapper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hf_to_gguf_attention_proj() {
        let v = remap(
            "model.layers.0.self_attn.q_proj.weight",
            NamingConvention::HuggingFace,
            NamingConvention::Gguf,
        );
        assert_eq!(
            v,
            RemapVerdict::Ok {
                mapped_name: "blk.0.attn_q.weight".into()
            }
        );
    }

    #[test]
    fn gguf_to_hf_attention_proj() {
        let v = remap(
            "blk.5.attn_v.weight",
            NamingConvention::Gguf,
            NamingConvention::HuggingFace,
        );
        assert_eq!(
            v,
            RemapVerdict::Ok {
                mapped_name: "model.layers.5.self_attn.v_proj.weight".into()
            }
        );
    }

    #[test]
    fn ffn_layer_remapped() {
        let v = remap(
            "model.layers.7.mlp.up_proj.weight",
            NamingConvention::HuggingFace,
            NamingConvention::Gguf,
        );
        assert_eq!(
            v,
            RemapVerdict::Ok {
                mapped_name: "blk.7.ffn_up.weight".into()
            }
        );
    }

    #[test]
    fn round_trip_preserves_name() {
        let original = "model.layers.3.self_attn.k_proj.weight";
        let mid = remap(
            original,
            NamingConvention::HuggingFace,
            NamingConvention::Gguf,
        );
        if let RemapVerdict::Ok { mapped_name } = mid {
            let back = remap(
                &mapped_name,
                NamingConvention::Gguf,
                NamingConvention::HuggingFace,
            );
            assert_eq!(
                back,
                RemapVerdict::Ok {
                    mapped_name: original.into()
                }
            );
        }
    }

    #[test]
    fn empty_name_invalid() {
        assert_eq!(
            remap("", NamingConvention::HuggingFace, NamingConvention::Gguf),
            RemapVerdict::InvalidSourceName
        );
    }

    #[test]
    fn same_convention_passes_through() {
        let v = remap(
            "blk.0.attn_q.weight",
            NamingConvention::Gguf,
            NamingConvention::Gguf,
        );
        assert_eq!(
            v,
            RemapVerdict::Ok {
                mapped_name: "blk.0.attn_q.weight".into()
            }
        );
    }

    #[test]
    fn onnx_unsupported() {
        assert_eq!(
            remap(
                "Constant_1234",
                NamingConvention::Onnx,
                NamingConvention::Gguf
            ),
            RemapVerdict::UnknownPattern
        );
    }

    #[test]
    fn unknown_role_rejected() {
        let v = remap(
            "model.layers.0.unknown_role.weight",
            NamingConvention::HuggingFace,
            NamingConvention::Gguf,
        );
        assert_eq!(v, RemapVerdict::UnknownPattern);
    }

    #[test]
    fn malformed_hf_rejected() {
        // Missing layer index.
        let v = remap(
            "model.layers.attn_q.weight",
            NamingConvention::HuggingFace,
            NamingConvention::Gguf,
        );
        assert_eq!(v, RemapVerdict::UnknownPattern);
    }
}
