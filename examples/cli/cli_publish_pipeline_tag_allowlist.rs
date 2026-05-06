//! # apr publish — `--pipeline-tag` Allowlist
//!
//! `apr publish ... --pipeline-tag <TAG>` accepts only HuggingFace's
//! enumerated pipeline tags. A typo or freeform tag breaks Hub search +
//! widget rendering on the model card. This recipe vendors the allowlist
//! and asserts the contract: unknown tags → reject, default tag is
//! "text-generation" (CLI default).
//!
//! Demonstrates the **PUBLISH.8** recipe for PMAT-098 (apr publish coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-PUB-001 + huggingface_hub PIPELINE_TAGS_LIST
//!
//! Run with: cargo run --example cli_publish_pipeline_tag_allowlist
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ALLOWED_PIPELINE_TAGS: &[&str] = &[
    "text-generation",
    "text-classification",
    "automatic-speech-recognition",
    "translation",
    "summarization",
    "image-classification",
    "image-to-text",
    "text-to-image",
    "feature-extraction",
    "fill-mask",
    "question-answering",
    "sentence-similarity",
    "token-classification",
    "zero-shot-classification",
    "audio-classification",
    "object-detection",
    "embeddings",
];

#[derive(Debug, PartialEq)]
pub enum PipelineTagVerdict {
    Ok,
    Unknown(String),
}

pub fn validate_pipeline_tag(tag: &str) -> PipelineTagVerdict {
    if ALLOWED_PIPELINE_TAGS.contains(&tag) {
        PipelineTagVerdict::Ok
    } else {
        PipelineTagVerdict::Unknown(tag.into())
    }
}

pub fn suggest_close_match(tag: &str) -> Option<&'static str> {
    let lower = tag.to_ascii_lowercase();
    ALLOWED_PIPELINE_TAGS
        .iter()
        .find(|allowed| allowed.contains(&*lower) || lower.contains(*allowed))
        .copied()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_publish_pipeline_tag_allowlist")?;

    let cases = [
        "text-generation",
        "automatic-speech-recognition",
        "embedding",      // typo (singular)
        "TextGeneration", // wrong case
        "image2text",     // wrong format
    ];
    for c in cases {
        let v = validate_pipeline_tag(c);
        match v {
            PipelineTagVerdict::Ok => println!("{c:>30}  →  OK"),
            PipelineTagVerdict::Unknown(_) => {
                let hint = suggest_close_match(c)
                    .map_or(String::new(), |s| format!(" (did you mean {s:?}?)"));
                println!("{c:>30}  →  Unknown{hint}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_text_generation_passes() {
        // The CLI default — must be valid.
        assert_eq!(
            validate_pipeline_tag("text-generation"),
            PipelineTagVerdict::Ok
        );
    }

    #[test]
    fn whisper_pipeline_passes() {
        assert_eq!(
            validate_pipeline_tag("automatic-speech-recognition"),
            PipelineTagVerdict::Ok
        );
    }

    #[test]
    fn typo_rejected_with_unknown() {
        assert!(matches!(
            validate_pipeline_tag("embedding"),
            PipelineTagVerdict::Unknown(_)
        ));
    }

    #[test]
    fn case_sensitive_validation() {
        // HF tags are canonically lowercase-with-hyphens.
        assert!(matches!(
            validate_pipeline_tag("Text-Generation"),
            PipelineTagVerdict::Unknown(_)
        ));
    }

    #[test]
    fn suggest_match_finds_substring() {
        // "embedding" should suggest "embeddings".
        assert_eq!(suggest_close_match("embedding"), Some("embeddings"));
    }

    #[test]
    fn suggest_match_returns_none_for_unrelated() {
        assert!(suggest_close_match("xyzqrs").is_none());
    }
}
