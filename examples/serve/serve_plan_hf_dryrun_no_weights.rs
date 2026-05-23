//! # apr serve plan — HuggingFace Dry-Run (no weight download)
//!
//! `apr serve plan hf://org/repo` reads ONLY the ~2 KB `config.json` from a
//! HuggingFace repo (no weight tensors), validates the model architecture
//! (n_layers, hidden_size, vocab_size, etc.), and prints a deployment plan
//! without ever downloading the GB-scale `.safetensors` shards. Useful in CI
//! to validate a model can be served before paying the bandwidth.
//!
//! This recipe demonstrates the `config.json` schema validator that the
//! `apr serve plan` flow uses, with a synthetic config inline so the recipe
//! is offline-only per IIUR.
//!
//! Demonstrates the **SRV+.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace (2024). Hub Model Card + config.json conventions. https://huggingface.co/docs/hub/model-cards
//!
//! Run with: cargo run --example serve_plan_hf_dryrun_no_weights
//!
//! Added by PMAT-077 (expand-cookbooks: apr serve anthropic + plan hf://).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::Value;

const SAMPLE_QWEN_CONFIG: &str = r#"{
  "model_type": "qwen2",
  "architectures": ["Qwen2ForCausalLM"],
  "vocab_size": 151936,
  "hidden_size": 3584,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "torch_dtype": "bfloat16"
}"#;

/// Minimal deployment plan extracted from a HuggingFace `config.json`.
#[derive(Debug)]
struct DeploymentPlan {
    model_type: String,
    architecture: String,
    n_layers: u64,
    hidden_size: u64,
    vocab_size: u64,
    estimated_params_b: f64,
}

fn build_plan_from_config(json: &str) -> Result<DeploymentPlan> {
    let cfg: Value = serde_json::from_str(json).map_err(|e| {
        apr_cookbook::CookbookError::Validation(format!("config.json parse error: {e}"))
    })?;
    let model_type = cfg["model_type"]
        .as_str()
        .ok_or_else(|| {
            apr_cookbook::CookbookError::Validation("model_type missing from config.json".into())
        })?
        .to_string();
    let architecture = cfg["architectures"][0]
        .as_str()
        .ok_or_else(|| {
            apr_cookbook::CookbookError::Validation(
                "architectures[0] missing from config.json".into(),
            )
        })?
        .to_string();
    let n_layers = cfg["num_hidden_layers"].as_u64().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("num_hidden_layers missing".into())
    })?;
    let hidden_size = cfg["hidden_size"]
        .as_u64()
        .ok_or_else(|| apr_cookbook::CookbookError::Validation("hidden_size missing".into()))?;
    let vocab_size = cfg["vocab_size"]
        .as_u64()
        .ok_or_else(|| apr_cookbook::CookbookError::Validation("vocab_size missing".into()))?;
    // Rough order-of-magnitude param estimate for a transformer:
    // params ~= n_layers * (12 * hidden^2 + 13*hidden) + 2 * vocab * hidden
    let h = hidden_size as f64;
    let l = n_layers as f64;
    let v = vocab_size as f64;
    let params = l * (12.0 * h * h + 13.0 * h) + 2.0 * v * h;
    let estimated_params_b = params / 1e9;
    Ok(DeploymentPlan {
        model_type,
        architecture,
        n_layers,
        hidden_size,
        vocab_size,
        estimated_params_b,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serve_plan_hf_dryrun_no_weights")?;
    let plan = build_plan_from_config(SAMPLE_QWEN_CONFIG)?;
    println!("deployment plan from synthetic Qwen-style config.json (no weights downloaded):");
    println!("  model_type: {}", plan.model_type);
    println!("  architecture: {}", plan.architecture);
    println!("  layers: {}", plan.n_layers);
    println!("  hidden_size: {}", plan.hidden_size);
    println!("  vocab_size: {}", plan.vocab_size);
    println!("  estimated params: ~{:.2}B", plan.estimated_params_b);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn synthetic_config_yields_qwen2_architecture() {
        let plan = build_plan_from_config(SAMPLE_QWEN_CONFIG).unwrap();
        assert_eq!(plan.model_type, "qwen2");
        assert_eq!(plan.architecture, "Qwen2ForCausalLM");
        assert_eq!(plan.n_layers, 28);
    }

    #[test]
    fn missing_required_field_rejected() {
        let bad = r#"{"model_type": "x", "architectures": ["X"]}"#;
        assert!(build_plan_from_config(bad).is_err());
    }

    #[test]
    fn estimated_params_in_billions_for_7b_class() {
        // 28 layers × 3584 hidden × 151936 vocab ≈ 7B params order of magnitude
        let plan = build_plan_from_config(SAMPLE_QWEN_CONFIG).unwrap();
        assert!(
            plan.estimated_params_b > 4.0 && plan.estimated_params_b < 12.0,
            "expected ~7B param estimate, got {:.2}B",
            plan.estimated_params_b
        );
    }
}
