//! # Recipe: Showcase Gallery — 5 Deterministic Demos
//!
//! **Category**: advanced
//! **CLI Equivalent**: `apr showcase --list --category=all`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example showcase_gallery` exits 0
//! 2. [x] `cargo test --example showcase_gallery` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr showcase` gallery listing in-process
//! 10. [x] Unit tests cover demo enumeration, category filters, reproducibility
//!
//! ## Learning Objective
//! Demonstrates the `apr showcase` gallery: enumerates five flagship demos
//! (classifier, chatbot, vision, speech, retrieval), each with a deterministic
//! fake inference result. Illustrates the minimal surface a demo harness
//! needs: name, category, inputs, expected outputs, reproducible signature.
//!
//! ## Run Command
//! ```bash
//! cargo run --example showcase_gallery
//! ```
//!
//! ## References
//! - Abadi, M. et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning*. OSDI. arXiv:1605.08695

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemoCard {
    pub name: String,
    pub category: &'static str,
    pub input_summary: String,
    pub output_summary: String,
    pub signature: u64,
}

fn fnv1a_u64(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn gallery() -> Vec<DemoCard> {
    let demos: &[(&str, &str, &str, &str)] = &[
        (
            "sentiment_classifier",
            "nlp",
            "I loved the new movie!",
            "positive (0.93)",
        ),
        (
            "chat_assistant",
            "nlp",
            "How do I center a div?",
            "You can use CSS flexbox...",
        ),
        (
            "image_classifier",
            "vision",
            "golden_retriever.jpg (224x224)",
            "golden_retriever (0.88)",
        ),
        (
            "whisper_transcribe",
            "speech",
            "hello_world.wav (1.2s, 16kHz)",
            "Hello world.",
        ),
        (
            "vector_search",
            "retrieval",
            "query: quantization techniques",
            "[paper_a.pdf#§3, paper_b.pdf#§2]",
        ),
    ];

    demos
        .iter()
        .map(|(name, cat, inp, out)| DemoCard {
            name: (*name).into(),
            category: cat,
            input_summary: (*inp).into(),
            output_summary: (*out).into(),
            signature: fnv1a_u64(&format!("{}|{}|{}|{}", name, cat, inp, out)),
        })
        .collect()
}

pub fn filter_category<'a>(gallery: &'a [DemoCard], cat: &str) -> Vec<&'a DemoCard> {
    gallery.iter().filter(|d| d.category == cat).collect()
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("showcase_gallery")?;
    println!("=== Recipe: {} ===", ctx.name());

    let g = gallery();
    println!("Showcase gallery — {} demos", g.len());
    println!(
        "{:<24} {:<12} {:<34} {:<28} SIG",
        "DEMO", "CATEGORY", "INPUT", "OUTPUT"
    );
    println!("{}", "-".repeat(120));
    for d in &g {
        println!(
            "{:<24} {:<12} {:<34} {:<28} {:016x}",
            d.name, d.category, d.input_summary, d.output_summary, d.signature
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "n_demos": g.len(),
        "categories": ["nlp", "vision", "speech", "retrieval"],
        "demos": g.iter().map(|d| json!({
            "name": d.name,
            "category": d.category,
            "input_summary": d.input_summary,
            "output_summary": d.output_summary,
            "signature": format!("{:016x}", d.signature),
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("showcase-gallery.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_has_five_demos() {
        assert_eq!(gallery().len(), 5);
    }

    #[test]
    fn all_demos_have_nonempty_fields() {
        for d in gallery() {
            assert!(!d.name.is_empty());
            assert!(!d.input_summary.is_empty());
            assert!(!d.output_summary.is_empty());
        }
    }

    #[test]
    fn filter_nlp_returns_two() {
        let g = gallery();
        let nlp = filter_category(&g, "nlp");
        assert_eq!(nlp.len(), 2);
    }

    #[test]
    fn filter_unknown_returns_empty() {
        let g = gallery();
        let none = filter_category(&g, "quantum-telepathy");
        assert!(none.is_empty());
    }

    #[test]
    fn signatures_are_deterministic() {
        let g1 = gallery();
        let g2 = gallery();
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert_eq!(a.signature, b.signature);
        }
    }
}
