#![allow(unused_imports)]
//! Demo O: Multi-Modal CLIP Search - text-to-image and image-to-text semantic search.
//! QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. arXiv:2103.00020

use std::collections::HashMap;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo O: Multi-Modal CLIP Search ===\n");
    let mut index = ClipIndex::new(42);
    let texts: Vec<_> = [
        ("t1", "cat on couch", "animals"),
        ("t2", "sunset over ocean", "nature"),
        ("t3", "person riding bicycle", "sports"),
        ("t4", "pasta with sauce", "food"),
        ("t5", "city skyline", "urban"),
    ]
    .iter()
    .map(|(id, c, cat)| TextDocument::new(*id, *c).with_metadata("category", *cat))
    .collect();
    index.index_texts(&texts);
    let images: Vec<_> = [
        ("img_cat", 1),
        ("img_sunset", 2),
        ("img_bike", 3),
        ("img_food", 4),
        ("img_city", 5),
    ]
    .iter()
    .map(|(id, s)| ImageDocument::test_pattern(*id, 64, 64, *s))
    .collect();
    index.index_images(&images);
    println!(
        "{} text, {} image, {} total",
        index.count_by_modality(Modality::Text),
        index.count_by_modality(Modality::Image),
        index.len()
    );

    for (q, label) in [
        ("cat on furniture", "text-to-image"),
        ("nature landscape", "text-to-text"),
        ("outdoor activities", "cross-modal"),
    ] {
        println!("\n{label}: '{q}'");
        let results = index.search_by_text(q, 3);
        for (i, r) in results.iter().enumerate() {
            println!("  {}. {} [{:?}] {:.4}", i + 1, r.id, r.modality, r.score);
        }
    }
    println!("\n=== Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_normalize_and_similarity() {
        let e = Embedding::new(vec![3.0, 4.0], Modality::Text);
        assert!(e.normalized);
        assert!((e.vector.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-6);
        assert!((e.similarity(&e) - 1.0).abs() < 1e-6);
        let e2 = Embedding::new(vec![0.0, 1.0], Modality::Text);
        let e3 = Embedding::new(vec![1.0, 0.0], Modality::Text);
        assert!(e2.similarity(&e3).abs() < 1e-6);
        let e4 = Embedding::new(vec![-1.0, 0.0], Modality::Text);
        assert!((e3.similarity(&e4) + 1.0).abs() < 1e-6);
        assert_eq!(Embedding::new(vec![1.0; 512], Modality::Image).dim(), 512);
        let mut z = Embedding {
            vector: vec![0.0; 10],
            modality: Modality::Text,
            normalized: false,
        };
        z.normalize();
        assert!(!z.normalized);
    }

    #[test]
    fn test_text_and_image_documents() {
        let t = TextDocument::new("t1", "hello").with_metadata("k", "v");
        assert_eq!(t.metadata.get("k"), Some(&"v".into()));
        let img = ImageDocument::test_pattern("i1", 16, 16, 42);
        assert_eq!(img.pixels.len(), 16 * 16 * 3);
        let img2 = ImageDocument::new("i2", 32, 32, vec![]).with_metadata("c", "cat");
        assert_eq!(img2.metadata.get("c"), Some(&"cat".into()));
    }

    #[test]
    fn test_text_encoder() {
        let mut enc = TextEncoder::new(1000, 128, 42);
        let e = enc.encode("hello world");
        assert_eq!(e.dim(), EMBEDDING_DIM);
        assert!(e.normalized);
        assert_eq!(enc.encode("").dim(), EMBEDDING_DIM);
        let mut enc2 = TextEncoder::new(1000, 128, 42);
        assert_eq!(
            enc2.encode("test").vector,
            TextEncoder::new(1000, 128, 42).encode("test").vector
        );
        assert_ne!(enc.encode("hello").vector, enc.encode("world").vector);
    }

    #[test]
    fn test_image_encoder() {
        let enc = ImageEncoder::new(128, 64, 42);
        let e = enc.encode(&ImageDocument::test_pattern("t", 32, 32, 1));
        assert_eq!(e.dim(), EMBEDDING_DIM);
        assert!(e.normalized);
        let e2 =
            ImageEncoder::new(128, 64, 42).encode(&ImageDocument::test_pattern("t", 32, 32, 1));
        assert_eq!(e.vector, e2.vector);
        assert_ne!(
            e.vector,
            enc.encode(&ImageDocument::test_pattern("t2", 32, 32, 2))
                .vector
        );
        assert_eq!(
            enc.encode(&ImageDocument::new("s", 8, 8, vec![128; 192]))
                .dim(),
            EMBEDDING_DIM
        );
    }

    #[test]
    fn test_clip_index() {
        let mut idx = ClipIndex::new(42);
        assert!(idx.is_empty());
        idx.index_text(TextDocument::new("t1", "cat"));
        idx.index_text(TextDocument::new("t2", "dog"));
        idx.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));
        idx.index_image(ImageDocument::test_pattern("i2", 32, 32, 2));
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.count_by_modality(Modality::Text), 2);
        assert_eq!(idx.count_by_modality(Modality::Image), 2);
        let r = idx.search_by_text("cat", 2);
        assert_eq!(r.len(), 2);
        assert!(r[0].score >= r[1].score);
        let rf = idx.search_by_text_filtered("cat", 5, Modality::Image);
        assert!(rf.iter().all(|r| r.modality == Modality::Image));
    }

    #[test]
    fn test_search_by_image_and_metadata() {
        let mut idx = ClipIndex::new(42);
        idx.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));
        idx.index_image(ImageDocument::test_pattern("i2", 32, 32, 2));
        let r = idx.search_by_image(&ImageDocument::test_pattern("q", 32, 32, 1), 2);
        assert_eq!(r.len(), 2);
        assert!(r[0].score >= r[1].score);
        let mut idx2 = ClipIndex::new(42);
        idx2.index_text(TextDocument::new("t1", "hello").with_metadata("k", "v"));
        let r2 = idx2.search_by_text("hello", 1);
        assert_eq!(r2[0].metadata.get("k"), Some(&"v".into()));
    }

    #[test]
    fn test_contrastive_loss_and_metrics() {
        assert_eq!(contrastive_loss(&[], &[], 0.07), 0.0);
        let te = vec![Embedding::new(vec![1.0, 0.0], Modality::Text)];
        let ie = vec![Embedding::new(vec![1.0, 0.0], Modality::Image)];
        assert!(contrastive_loss(&te, &ie, 0.07) < 1.0);
        let mk = |id: &str, s: f32| SearchResult {
            id: id.into(),
            modality: Modality::Text,
            score: s,
            metadata: HashMap::new(),
        };
        let results = vec![mk("a", 0.9), mk("b", 0.8)];
        assert_eq!(recall_at_k(&results, "a", 1), 1.0);
        assert_eq!(recall_at_k(&results, "b", 1), 0.0);
        assert_eq!(recall_at_k(&results, "b", 2), 1.0);
        assert_eq!(mean_reciprocal_rank(&results, "a"), 1.0);
        assert_eq!(mean_reciprocal_rank(&results, "b"), 0.5);
        assert_eq!(mean_reciprocal_rank(&results, "x"), 0.0);
    }

    #[test]
    fn test_full_pipeline() {
        let mut idx = ClipIndex::new(42);
        idx.index_text(TextDocument::new("d1", "a cute cat").with_metadata("type", "desc"));
        idx.index_image(ImageDocument::test_pattern("i1", 32, 32, 100).with_metadata("s", "cat"));
        let r = idx.search_by_text("cat", 2);
        assert_eq!(r.len(), 2);
        let mods: Vec<_> = r.iter().map(|r| r.modality).collect();
        assert!(mods.contains(&Modality::Text) && mods.contains(&Modality::Image));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_embedding_normalized(vec in prop::collection::vec(-10.0f32..10.0f32, 10..100)) {
            if vec.iter().any(|&x| x.abs() > 1e-8) {
                let e = Embedding::new(vec, Modality::Text);
                prop_assert!((e.vector.iter().map(|x| x*x).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn prop_similarity_symmetric_and_bounded(v1 in prop::collection::vec(-1.0f32..1.0, 10..20), v2 in prop::collection::vec(-1.0f32..1.0, 10..20)) {
            let len = v1.len().min(v2.len());
            let (e1, e2) = (Embedding::new(v1[..len].to_vec(), Modality::Text), Embedding::new(v2[..len].to_vec(), Modality::Text));
            prop_assert!((e1.similarity(&e2) - e2.similarity(&e1)).abs() < 1e-6);
            let s = e1.similarity(&e2);
            prop_assert!(s >= -1.0 - 1e-6 && s <= 1.0 + 1e-6);
        }

        #[test]
        fn prop_encoder_output_dim(text in "[a-z ]{1,50}") {
            prop_assert_eq!(TextEncoder::new(1000, 128, 42).encode(&text).dim(), EMBEDDING_DIM);
        }

        #[test]
        fn prop_image_encoder_dim(w in 8u32..64, h in 8u32..64, seed in 1u32..1000) {
            prop_assert_eq!(ImageEncoder::new(128, 64, 42).encode(&ImageDocument::test_pattern("t", w, h, seed)).dim(), EMBEDDING_DIM);
        }

        #[test]
        fn prop_index_count(nt in 0usize..10, ni in 0usize..10) {
            let mut idx = ClipIndex::new(42);
            for i in 0..nt { idx.index_text(TextDocument::new(format!("t{i}"), format!("text {i}"))); }
            for i in 0..ni { idx.index_image(ImageDocument::test_pattern(format!("i{i}"), 32, 32, i as u32)); }
            prop_assert_eq!(idx.len(), nt + ni);
        }

        #[test]
        fn prop_search_ordered(seed in 1u64..1000, nd in 3usize..10) {
            let mut idx = ClipIndex::new(seed);
            for i in 0..nd { idx.index_text(TextDocument::new(format!("t{i}"), format!("doc {i}"))); }
            let r = idx.search_by_text("doc", nd);
            for i in 1..r.len() { prop_assert!(r[i-1].score >= r[i].score); }
        }
    }
}
