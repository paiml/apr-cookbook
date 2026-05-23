#![allow(unused_imports)]
//! Demo K: RAG Pipeline - chunking, embedding, vector search, context injection.
//! QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401

use std::collections::HashMap;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo K: RAG Pipeline ===\n");
    let mut pipeline = RagPipeline::new();
    let corpus = generate_sample_corpus(20, 42);
    let n = pipeline.ingest_batch(&corpus).expect("ingest");
    let s = pipeline.stats();
    println!(
        "{} docs, {n} chunks indexed, {} dims",
        s.document_count, s.embedding_dim
    );
    for q in [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain transformers",
    ] {
        let r = pipeline.query(q, 3).expect("query");
        println!("\nQ: {q}");
        for (i, sr) in r.results.iter().take(3).enumerate() {
            println!(
                "  {}. [{:.3}] {}...",
                i + 1,
                sr.score,
                &sr.chunk.content[..sr.chunk.content.len().min(50)]
            );
        }
    }
    println!("\n=== Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_and_metadata() {
        let doc = Document::new("t", "Hello world").with_metadata("k", "v");
        assert_eq!(doc.token_count(), 2);
        assert_eq!(doc.metadata.get("k"), Some(&"v".to_string()));
    }

    #[test]
    fn test_chunker_all_strategies() {
        let doc = Document::new("t", "ABCDEFGHIJ");
        let c = Chunker::new(ChunkingStrategy::FixedSize {
            chunk_size: 4,
            overlap: 1,
        })
        .chunk(&doc)
        .unwrap();
        assert!(!c.is_empty());
        assert_eq!(c[0].content, "ABCD");

        let doc2 = Document::new("t", "First. Second. Third.");
        assert!(
            !Chunker::new(ChunkingStrategy::Sentence { max_sentences: 2 })
                .chunk(&doc2)
                .unwrap()
                .is_empty()
        );

        let doc3 = Document::new("t", "Para 1\n\nPara 2\n\nPara 3");
        assert_eq!(
            Chunker::new(ChunkingStrategy::Paragraph)
                .chunk(&doc3)
                .unwrap()
                .len(),
            3
        );

        assert!(Chunker::default().chunk(&Document::new("t", "")).is_err());
    }

    #[test]
    fn test_embedding_creation_and_dim() {
        assert!(Embedding::new(vec![0.1; EMBEDDING_DIM], "t".into()).is_ok());
        assert!(Embedding::new(vec![0.1; 10], "t".into()).is_err());
    }

    #[test]
    fn test_embedding_similarity() {
        let mut v1 = vec![0.0_f32; EMBEDDING_DIM];
        v1[0] = 1.0;
        let mut v2 = vec![0.0_f32; EMBEDDING_DIM];
        v2[0] = 1.0;
        let mut v3 = vec![0.0_f32; EMBEDDING_DIM];
        v3[1] = 1.0;
        let e1 = Embedding::new(v1, "a".into()).unwrap();
        let e2 = Embedding::new(v2, "b".into()).unwrap();
        let e3 = Embedding::new(v3, "c".into()).unwrap();
        assert!((e1.cosine_similarity(&e2) - 1.0).abs() < 0.01);
        assert!(e1.cosine_similarity(&e3).abs() < 0.01);
    }

    #[test]
    fn test_embedding_model() {
        let m = EmbeddingModel::default();
        let e = m.embed("Hello world").unwrap();
        assert_eq!(e.values.len(), EMBEDDING_DIM);
        assert_eq!(e.values, m.embed("Hello world").unwrap().values);
        assert!(m.embed("").is_err());
    }

    #[test]
    fn test_vector_index() {
        let mut idx = VectorIndex::new("t");
        let m = EmbeddingModel::default();
        for i in 0..5 {
            let c = Chunk {
                doc_id: format!("d{i}"),
                chunk_index: 0,
                content: format!("Content {i}"),
                start_offset: 0,
                end_offset: 10,
            };
            idx.add(c, m.embed(&format!("Content {i}")).unwrap());
        }
        let qe = m.embed("Content 0").unwrap();
        let r = idx.search(&qe, 3);
        assert_eq!(r.len(), 3);
        assert!(r[0].score >= r[1].score);
    }

    #[test]
    fn test_vector_index_remove() {
        let mut idx = VectorIndex::new("t");
        let m = EmbeddingModel::default();
        for i in 0..3 {
            let c = Chunk {
                doc_id: "d1".into(),
                chunk_index: i,
                content: format!("Chunk {i}"),
                start_offset: 0,
                end_offset: 7,
            };
            idx.add(c, m.embed(&format!("Chunk {i}")).unwrap());
        }
        assert_eq!(idx.remove_document("d1"), 3);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_pipeline_ingest_query_stats() {
        let mut p = RagPipeline::new();
        let docs = generate_sample_corpus(5, 42);
        p.ingest_batch(&docs).unwrap();
        let r = p.query("machine learning", 3).unwrap();
        assert!(!r.results.is_empty());
        let s = p.stats();
        assert!(s.chunk_count > 0);
        assert_eq!(s.document_count, 5);
    }

    #[test]
    fn test_query_result_avg_score() {
        let mk = |s: f32| SearchResult {
            chunk: Chunk {
                doc_id: "d".into(),
                chunk_index: 0,
                content: "c".into(),
                start_offset: 0,
                end_offset: 1,
            },
            score: s,
            index_id: 0,
        };
        let r = QueryResult {
            query: "t".into(),
            results: vec![mk(0.8), mk(0.6)],
            context: "t".into(),
        };
        assert!((r.avg_score() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_sample_corpus() {
        let c = generate_sample_corpus(10, 42);
        assert_eq!(c.len(), 10);
        assert_eq!(c[0].id, "doc_0");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_chunker_non_empty(content in "[a-z]{10,100}") {
            let doc = Document::new("t", &content);
            let chunks = Chunker::default().chunk(&doc);
            prop_assert!(chunks.is_ok());
            prop_assert!(!chunks.unwrap().is_empty());
        }

        #[test]
        fn prop_embedding_normalized(text in "[a-z]{1,5}[a-z ]{4,45}") {
            if let Ok(e) = EmbeddingModel::default().embed(&text) {
                let norm: f32 = e.values.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 0.01);
            }
        }

        #[test]
        fn prop_cosine_similarity_bounds(seed1 in 0u64..1000) {
            let m = EmbeddingModel::new(EMBEDDING_DIM, seed1);
            let sim = m.embed("text one").unwrap().cosine_similarity(&m.embed("text two").unwrap());
            prop_assert!(sim >= -1.0 && sim <= 1.0);
        }

        #[test]
        fn prop_self_similarity_is_one(text in "[a-z]{1,5}[a-z ]{4,25}") {
            if let Ok(emb) = EmbeddingModel::default().embed(&text) {
                prop_assert!((emb.cosine_similarity(&emb) - 1.0).abs() < 0.01);
            }
        }

        #[test]
        fn prop_search_returns_at_most_k(k in 1usize..10) {
            let mut idx = VectorIndex::new("t");
            let m = EmbeddingModel::default();
            for i in 0..5 {
                let c = Chunk { doc_id: format!("d{i}"), chunk_index: 0, content: format!("Content {i}"), start_offset: 0, end_offset: 10 };
                idx.add(c, m.embed(&format!("Content {i}")).unwrap());
            }
            prop_assert!(idx.search(&m.embed("query").unwrap(), k).len() <= k);
        }

        #[test]
        fn prop_pipeline_ingest_count(n in 1usize..10) {
            let mut p = RagPipeline::new();
            p.ingest_batch(&generate_sample_corpus(n, 42)).unwrap();
            prop_assert_eq!(p.stats().document_count, n);
        }
    }
}
