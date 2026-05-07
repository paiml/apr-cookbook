//! # Architecture Detector — Cross-Family Discriminator Dispatch
//!
//! Take any HF `config.json` and identify which architecture family it
//! belongs to by inspecting the **discriminator fields** each individual
//! family recipe surfaces (`rope_theta`, `head_dim`, `tie_word_embeddings`,
//! `qkv_proj_fused`, `query_pre_attn_scalar`, `n_embd`, `use_parallel_residual`,
//! `n_routed_experts`, `mamba_d_state`, `time_mix_extra_dim`,
//! `do_layer_norm_before`, `state_size`, `type_vocab_size`,
//! `ffn_multipliers`, `sliding_window`).
//!
//! Demonstrates the **ARCH-DETECTOR** recipe per
//! `docs/specifications/architecture-demos.md` v1.1.
//!
//! ## Upstream contribution
//!
//! This recipe's dispatch logic was lifted upstream into
//! `aprender::format::FamilyRegistry::detect_from_config_str` (aprender PR
//! [#1562](https://github.com/paiml/aprender/pull/1562)). When apr-cookbook
//! bumps its aprender pin to a release containing #1562, this recipe will
//! be refactored to call the upstream API directly:
//!
//! ```ignore
//! use aprender::format::FamilyRegistry;
//! let family = FamilyRegistry::detect_from_config_str(&body); // Option<&'static str>
//! ```
//!
//! Until then, the body of this recipe is the reference implementation
//! the upstream API mirrors verbatim — same priority order, same
//! discriminator catalog. The 22 unit tests serve as a falsification
//! contract on the upstream behavior.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-detector-v1.yaml (grade C; lean_status: wip)
//! Citation: HuggingFace Transformers `config.json` schema; family-discriminator catalog from architecture-demos.md
//!
//! Run with: cargo run --example inference_arch_detector
//!
//! Added by PMAT-309 (architecture-demos v1.1: cross-family detector).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedFamily {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    Qwen3_5,
    Phi,
    Gemma,
    Gpt2,
    GptNeox,
    Deepseek,
    FalconH1,
    Rwkv7,
    Openelm,
    Opt,
    Mamba,
    Bert,
    Whisper,
    Moonshine,
}

impl DetectedFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3_5 => "qwen3_5",
            Self::Phi => "phi",
            Self::Gemma => "gemma",
            Self::Gpt2 => "gpt2",
            Self::GptNeox => "gptneox",
            Self::Deepseek => "deepseek",
            Self::FalconH1 => "falcon_h1",
            Self::Rwkv7 => "rwkv7",
            Self::Openelm => "openelm",
            Self::Opt => "opt",
            Self::Mamba => "mamba",
            Self::Bert => "bert",
            Self::Whisper => "whisper",
            Self::Moonshine => "moonshine",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DetectorVerdict {
    Family {
        family: DetectedFamily,
        match_reason: String,
    },
    UnknownFamily {
        config_excerpt: String,
    },
    InvalidFixture,
}

pub fn detect(fixture_path: &str) -> DetectorVerdict {
    if !std::path::Path::new(fixture_path).exists() {
        return DetectorVerdict::InvalidFixture;
    }
    let Ok(body) = std::fs::read_to_string(fixture_path) else {
        return DetectorVerdict::InvalidFixture;
    };
    detect_from_str(&body)
}

/// Order matters: more-specific discriminators first. e.g., qwen3_5 has
/// tie_word_embeddings AND head_dim, so it must be checked before qwen3.
pub fn detect_from_str(body: &str) -> DetectorVerdict {
    // Qwen3.5: explicit tie_word_embeddings + head_dim (extends Qwen3)
    if body.contains("tie_word_embeddings") && body.contains("head_dim") && body.contains("qwen3_5")
    {
        return ok(
            DetectedFamily::Qwen3_5,
            "tie_word_embeddings + head_dim + qwen3_5",
        );
    }
    // Qwen3: head_dim explicit, no tie_word_embeddings
    if body.contains("head_dim") && body.contains("qwen3") && !body.contains("qwen3_5") {
        return ok(DetectedFamily::Qwen3, "head_dim + qwen3");
    }
    // Qwen2: rope_theta=1000000 + qwen2 model_type, OR qkv biases pattern
    if body.contains("qwen2") && body.contains("rope_theta") {
        return ok(
            DetectedFamily::Qwen2,
            "qwen2 model_type + rope_theta=1000000",
        );
    }
    // Phi: qkv_proj_fused field
    if body.contains("qkv_proj_fused") {
        return ok(DetectedFamily::Phi, "qkv_proj_fused field present");
    }
    // Gemma: query_pre_attn_scalar
    if body.contains("query_pre_attn_scalar") {
        return ok(DetectedFamily::Gemma, "query_pre_attn_scalar field present");
    }
    // GPT-NeoX: use_parallel_residual
    if body.contains("use_parallel_residual") {
        return ok(
            DetectedFamily::GptNeox,
            "use_parallel_residual field present",
        );
    }
    // OPT: do_layer_norm_before
    if body.contains("do_layer_norm_before") {
        return ok(DetectedFamily::Opt, "do_layer_norm_before field present");
    }
    // GPT-2: n_embd short-name field
    if body.contains("\"n_embd\"") {
        return ok(DetectedFamily::Gpt2, "n_embd short-name field present");
    }
    // OpenELM: ffn_multipliers + num_query_heads arrays
    if body.contains("ffn_multipliers") && body.contains("num_query_heads") {
        return ok(
            DetectedFamily::Openelm,
            "ffn_multipliers + num_query_heads arrays",
        );
    }
    // DeepSeek: n_routed_experts MoE field
    if body.contains("n_routed_experts") {
        return ok(DetectedFamily::Deepseek, "n_routed_experts MoE field");
    }
    // Falcon-H1: mamba_d_state + mamba_expand (hybrid SSM)
    if body.contains("mamba_d_state") && body.contains("mamba_expand") && body.contains("falcon_h1")
    {
        return ok(
            DetectedFamily::FalconH1,
            "mamba_d_state + mamba_expand + falcon_h1",
        );
    }
    // RWKV-7: time_mix_extra_dim
    if body.contains("time_mix_extra_dim") {
        return ok(
            DetectedFamily::Rwkv7,
            "time_mix_extra_dim linear-attention field",
        );
    }
    // MAMBA: state_size + conv_kernel (pure SSM, no transformer fields)
    if body.contains("state_size")
        && body.contains("conv_kernel")
        && !body.contains("num_attention_heads")
    {
        return ok(
            DetectedFamily::Mamba,
            "state_size + conv_kernel without attention",
        );
    }
    // BERT: type_vocab_size encoder-only
    if body.contains("type_vocab_size") {
        return ok(DetectedFamily::Bert, "type_vocab_size encoder-only field");
    }
    // Mistral: sliding_window field (without Qwen2's qwen2 marker)
    if body.contains("sliding_window") && body.contains("MistralForCausalLM") {
        return ok(
            DetectedFamily::Mistral,
            "sliding_window + MistralForCausalLM architecture",
        );
    }
    // Whisper: WhisperForConditionalGeneration
    if body.contains("WhisperForConditionalGeneration") {
        return ok(
            DetectedFamily::Whisper,
            "WhisperForConditionalGeneration architecture",
        );
    }
    // Moonshine: MoonshineForConditionalGeneration
    if body.contains("MoonshineForConditionalGeneration") {
        return ok(
            DetectedFamily::Moonshine,
            "MoonshineForConditionalGeneration architecture",
        );
    }
    // Llama: LlamaForCausalLM as the catch-all for transformer configs that
    // didn't match anything more specific. Must be checked LAST.
    if body.contains("LlamaForCausalLM") || body.contains("\"model_type\": \"llama\"") {
        return ok(
            DetectedFamily::Llama,
            "LlamaForCausalLM (catch-all transformer)",
        );
    }
    DetectorVerdict::UnknownFamily {
        config_excerpt: body.chars().take(120).collect(),
    }
}

fn ok(family: DetectedFamily, reason: &str) -> DetectorVerdict {
    DetectorVerdict::Family {
        family,
        match_reason: reason.to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_detector")?;
    for fam in [
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "phi",
        "gemma",
        "gpt2",
        "gptneox",
        "deepseek",
        "falcon_h1",
        "rwkv7",
        "openelm",
        "opt",
        "mamba",
        "bert",
    ] {
        let path = format!("tests/fixtures/architectures/{fam}/config.json");
        match detect(&path) {
            DetectorVerdict::Family {
                family,
                match_reason,
            } => {
                println!("{fam:>12}: detected {} ({match_reason})", family.as_str());
            }
            other => println!("{fam:>12}: {other:?}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(family: &str) -> String {
        format!("tests/fixtures/architectures/{family}/config.json")
    }

    fn detect_family(family: &str) -> Option<DetectedFamily> {
        match detect(&fixture(family)) {
            DetectorVerdict::Family { family, .. } => Some(family),
            _ => None,
        }
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_returns_invalid() {
        assert_eq!(detect("/no/such/path"), DetectorVerdict::InvalidFixture);
    }

    #[test]
    fn unknown_config_returns_unknown_family() {
        let body = r#"{"model_type": "unknown_arch"}"#;
        assert!(matches!(
            detect_from_str(body),
            DetectorVerdict::UnknownFamily { .. }
        ));
    }

    #[test]
    fn detects_llama() {
        assert_eq!(detect_family("llama"), Some(DetectedFamily::Llama));
    }

    #[test]
    fn detects_mistral() {
        assert_eq!(detect_family("mistral"), Some(DetectedFamily::Mistral));
    }

    #[test]
    fn detects_qwen2() {
        assert_eq!(detect_family("qwen2"), Some(DetectedFamily::Qwen2));
    }

    #[test]
    fn detects_qwen3() {
        assert_eq!(detect_family("qwen3"), Some(DetectedFamily::Qwen3));
    }

    #[test]
    fn detects_qwen3_5() {
        assert_eq!(detect_family("qwen3_5"), Some(DetectedFamily::Qwen3_5));
    }

    #[test]
    fn detects_phi() {
        assert_eq!(detect_family("phi"), Some(DetectedFamily::Phi));
    }

    #[test]
    fn detects_gemma() {
        assert_eq!(detect_family("gemma"), Some(DetectedFamily::Gemma));
    }

    #[test]
    fn detects_gpt2() {
        assert_eq!(detect_family("gpt2"), Some(DetectedFamily::Gpt2));
    }

    #[test]
    fn detects_gptneox() {
        assert_eq!(detect_family("gptneox"), Some(DetectedFamily::GptNeox));
    }

    #[test]
    fn detects_deepseek() {
        assert_eq!(detect_family("deepseek"), Some(DetectedFamily::Deepseek));
    }

    #[test]
    fn detects_falcon_h1() {
        assert_eq!(detect_family("falcon_h1"), Some(DetectedFamily::FalconH1));
    }

    #[test]
    fn detects_rwkv7() {
        assert_eq!(detect_family("rwkv7"), Some(DetectedFamily::Rwkv7));
    }

    #[test]
    fn detects_openelm() {
        assert_eq!(detect_family("openelm"), Some(DetectedFamily::Openelm));
    }

    #[test]
    fn detects_opt() {
        assert_eq!(detect_family("opt"), Some(DetectedFamily::Opt));
    }

    #[test]
    fn detects_mamba() {
        assert_eq!(detect_family("mamba"), Some(DetectedFamily::Mamba));
    }

    #[test]
    fn detects_bert() {
        assert_eq!(detect_family("bert"), Some(DetectedFamily::Bert));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = detect(&fixture("llama"));
        let b = detect(&fixture("llama"));
        assert_eq!(a, b);
    }

    #[test]
    fn family_as_str_round_trip() {
        for fam in [
            DetectedFamily::Llama,
            DetectedFamily::Mistral,
            DetectedFamily::Qwen2,
            DetectedFamily::Phi,
            DetectedFamily::Bert,
        ] {
            assert!(!fam.as_str().is_empty());
            assert!(!fam.as_str().contains(' '));
        }
    }

    #[test]
    fn match_reason_nonempty_for_each_family() {
        for fam in [
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3_5",
            "phi",
            "gemma",
            "gpt2",
            "gptneox",
            "deepseek",
            "falcon_h1",
            "rwkv7",
            "openelm",
            "opt",
            "mamba",
            "bert",
        ] {
            if let DetectorVerdict::Family { match_reason, .. } = detect(&fixture(fam)) {
                assert!(
                    !match_reason.is_empty(),
                    "family {fam} has empty match_reason"
                );
            } else {
                panic!("expected Family verdict for {fam}");
            }
        }
    }
}
