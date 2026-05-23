//! # apr validate-manifest --artifact model.safetensors (FALSIFY-PM-007)
//!
//! `apr validate-manifest --artifact model.safetensors` parses the
//! safetensors header JSON and verifies per-tensor dtype matches
//! `manifest.quantization`. Weight tensors must match (fp16→F16,
//! bf16→BF16, fp32→F32); norm/bias tensors may stay F32. This recipe
//! demonstrates the canary that catches the SHIP-TWO-001 §12.7.2
//! ship-blocker: a 30.46 GiB F32 file accidentally manifested as fp16.
//!
//! Demonstrates the **CLI+.4** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: publish-manifest-v1.yaml v1.1.0 FALSIFY-PM-007 (SafeTensors header dtype canary)
//!
//! Run with: cargo run --example cli_validate_manifest_safetensors_dtype
//!
//! Added by PMAT-076 (expand-cookbooks: apr publish end-to-end).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

/// Per-tensor dtype check against manifest.quantization. Returns Err if any
/// non-norm/non-bias weight tensor has a dtype that contradicts the
/// manifest's declared quantization.
fn validate_safetensors_dtype(header: &Value, manifest_quant: &str) -> Result<()> {
    let expected = match manifest_quant {
        "fp16" => "F16",
        "bf16" => "BF16",
        "fp32" => "F32",
        other => {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "FALSIFY-PM-007: unknown quantization `{other}`"
            )))
        }
    };
    let tensors = header.as_object().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("safetensors header must be a JSON object".into())
    })?;
    for (name, meta) in tensors {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().unwrap_or("?");
        // Norm and bias tensors are allowed to stay F32 even when weights are quantized.
        let is_exempt = name.contains("norm") || name.contains("bias");
        if is_exempt {
            continue;
        }
        if dtype != expected {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "FALSIFY-PM-007: tensor `{name}` has dtype `{dtype}`, expected `{expected}` for quantization `{manifest_quant}` (would have caught the SHIP-TWO-001 §12.7.2 30.46 GiB F32 fp16-manifest bug)"
            )));
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_manifest_safetensors_dtype")?;

    // Synthetic safetensors header for a small fp16 model.
    let header_good = json!({
        "__metadata__": {"format": "pt"},
        "model.layers.0.attention.q_proj.weight": {"dtype": "F16", "shape": [4096, 4096]},
        "model.layers.0.attention.norm.weight": {"dtype": "F32", "shape": [4096]},  // norm exempt
        "model.layers.0.mlp.gate_proj.weight": {"dtype": "F16", "shape": [4096, 11008]},
        "model.embed_tokens.weight": {"dtype": "F16", "shape": [151936, 4096]}
    });

    validate_safetensors_dtype(&header_good, "fp16")?;
    println!("good fp16 manifest passes FALSIFY-PM-007 check");

    // The exact ship-blocker from SHIP-TWO-001 §12.7.2: F32 weights claimed as fp16.
    let header_bad = json!({
        "model.layers.0.attention.q_proj.weight": {"dtype": "F32", "shape": [4096, 4096]}
    });
    let err = validate_safetensors_dtype(&header_bad, "fp16");
    assert!(err.is_err(), "ship-blocker scenario should fail validation");
    println!(
        "ship-blocker scenario correctly rejected: {}",
        err.unwrap_err()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_check_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fp16_weights_with_norm_f32_passes() {
        let h = json!({
            "model.weight": {"dtype": "F16", "shape": [10]},
            "model.norm.weight": {"dtype": "F32", "shape": [10]},
            "model.bias": {"dtype": "F32", "shape": [10]}
        });
        assert!(validate_safetensors_dtype(&h, "fp16").is_ok());
    }

    #[test]
    fn ship_two_001_30gb_f32_fp16_manifest_bug_caught() {
        // The exact bug from SHIP-TWO-001 §12.7.2: a 30.46 GiB F32 weight
        // accidentally published with `quantization: fp16` in the manifest.
        let h = json!({
            "model.weight": {"dtype": "F32", "shape": [4096, 4096]}
        });
        let err = validate_safetensors_dtype(&h, "fp16");
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("FALSIFY-PM-007"),
            "expected FALSIFY-PM-007 in error: {msg}"
        );
    }

    #[test]
    fn unknown_quantization_rejected() {
        let h = json!({"a": {"dtype": "F16", "shape": [1]}});
        assert!(validate_safetensors_dtype(&h, "int4").is_err());
    }
}
