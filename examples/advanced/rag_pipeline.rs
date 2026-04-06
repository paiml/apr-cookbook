//! Demo K: RAG Pipeline - chunking, embedding, vector search, context injection.
//! QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//! ## References
//! - Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401

use std::collections::HashMap;

pub const EMBEDDING_DIM: usize = 384;
pub const MAX_CONTEXT_TOKENS: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub enum RagError {
    EmptyDocument,
    InvalidEmbeddingDim { expected: usize, got: usize },
    IndexNotFound(String),
    DocumentNotFound(String),
    ChunkingError(String),
    EmbeddingError(String),
    SearchError(String),
    ContextOverflow { max_tokens: usize, required: usize },
}

impl std::fmt::Display for RagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocument => write!(f, "Empty document"),
            Self::InvalidEmbeddingDim { expected, got } => {
                write!(f, "Embedding dim: expected {expected}, got {got}")
            }
            Self::IndexNotFound(n) => write!(f, "Index not found: {n}"),
            Self::DocumentNotFound(id) => write!(f, "Document not found: {id}"),
            Self::ChunkingError(m) | Self::EmbeddingError(m) | Self::SearchError(m) => {
                write!(f, "{m}")
            }
            Self::ContextOverflow {
                max_tokens,
                required,
            } => write!(f, "Context overflow: max {max_tokens}, need {required}"),
        }
    }
}

impl std::error::Error for RagError {}

pub type Result<T> = std::result::Result<T, RagError>;

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    pub fn token_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub chunk_index: usize,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
}

impl Chunk {
    pub fn token_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    FixedSize { chunk_size: usize, overlap: usize },
    Sentence { max_sentences: usize },
    Paragraph,
    RecursiveToken { chunk_size: usize, overlap: usize },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::RecursiveToken {
            chunk_size: 512,
            overlap: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunker {
    strategy: ChunkingStrategy,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkingStrategy::default())
    }
}

impl Chunker {
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    pub fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        if doc.content.is_empty() {
            return Err(RagError::EmptyDocument);
        }
        match self.strategy {
            ChunkingStrategy::FixedSize {
                chunk_size,
                overlap,
            } => self.chunk_fixed(doc, chunk_size, overlap),
            ChunkingStrategy::Sentence { max_sentences } => self.chunk_sentence(doc, max_sentences),
            ChunkingStrategy::Paragraph => self.chunk_paragraph(doc),
            ChunkingStrategy::RecursiveToken {
                chunk_size,
                overlap,
            } => self.chunk_recursive(doc, chunk_size, overlap),
        }
    }

    fn chunk_fixed(&self, doc: &Document, size: usize, overlap: usize) -> Result<Vec<Chunk>> {
        let chars: Vec<char> = doc.content.chars().collect();
        let (mut chunks, mut start, mut idx) = (Vec::new(), 0, 0);
        while start < chars.len() {
            let end = (start + size).min(chars.len());
            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index: idx,
                content: chars[start..end].iter().collect(),
                start_offset: start,
                end_offset: end,
            });
            idx += 1;
            start = if end == chars.len() {
                chars.len()
            } else {
                end.saturating_sub(overlap)
            };
        }
        Ok(chunks)
    }

    fn chunk_sentence(&self, doc: &Document, max: usize) -> Result<Vec<Chunk>> {
        let sentences: Vec<&str> = doc
            .content
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();
        let mut chunks = Vec::new();
        let mut offset = 0;
        for (i, group) in sentences.chunks(max).enumerate() {
            let content = group.join(". ").trim().to_string() + ".";
            let end = offset + content.len();
            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index: i,
                content: content.clone(),
                start_offset: offset,
                end_offset: end,
            });
            offset = end;
        }
        Ok(chunks)
    }

    fn chunk_paragraph(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let paras: Vec<&str> = doc
            .content
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .collect();
        let mut chunks = Vec::new();
        let mut offset = 0;
        for (i, para) in paras.iter().enumerate() {
            let content = para.trim().to_string();
            let end = offset + content.len();
            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index: i,
                content,
                start_offset: offset,
                end_offset: end,
            });
            offset = end + 2;
        }
        Ok(chunks)
    }

    fn chunk_recursive(&self, doc: &Document, size: usize, overlap: usize) -> Result<Vec<Chunk>> {
        let words: Vec<&str> = doc.content.split_whitespace().collect();
        let (mut chunks, mut start, mut idx, mut char_off) = (Vec::new(), 0, 0, 0);
        while start < words.len() {
            let end = (start + size).min(words.len());
            let content = words[start..end].join(" ");
            let clen = content.len();
            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index: idx,
                content,
                start_offset: char_off,
                end_offset: char_off + clen,
            });
            char_off += clen + 1;
            idx += 1;
            start = if end == words.len() {
                words.len()
            } else {
                end.saturating_sub(overlap)
            };
        }
        Ok(chunks)
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub source_text: String,
}

impl Embedding {
    pub fn new(values: Vec<f32>, source_text: String) -> Result<Self> {
        if values.len() != EMBEDDING_DIM {
            return Err(RagError::InvalidEmbeddingDim {
                expected: EMBEDDING_DIM,
                got: values.len(),
            });
        }
        Ok(Self {
            values,
            source_text,
        })
    }
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
    pub fn l2_distance(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    dim: usize,
    seed: u64,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::new(EMBEDDING_DIM, 42)
    }
}

impl EmbeddingModel {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    pub fn embed(&self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(RagError::EmbeddingError("Empty text".into()));
        }
        let mut values = vec![0.0_f32; self.dim];
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let hash = self.hash_word(word, i as u64);
            for (j, v) in values.iter_mut().enumerate() {
                *v += (hash.wrapping_add(j as u64) as f32 * 0.0001).sin() / words.len() as f32;
            }
        }
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut values {
                *v /= norm;
            }
        }
        Embedding::new(values, text.to_string())
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn hash_word(&self, word: &str, position: u64) -> u64 {
        let mut hash = self.seed.wrapping_add(position);
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }
}

#[derive(Debug, Clone)]
struct IndexedChunk {
    chunk: Chunk,
    embedding: Embedding,
    index_id: usize,
}

#[derive(Debug)]
pub struct VectorIndex {
    pub name: String,
    chunks: Vec<IndexedChunk>,
    doc_chunks: HashMap<String, Vec<usize>>,
    next_id: usize,
}

impl VectorIndex {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            chunks: Vec::new(),
            doc_chunks: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add(&mut self, chunk: Chunk, embedding: Embedding) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.doc_chunks
            .entry(chunk.doc_id.clone())
            .or_default()
            .push(id);
        self.chunks.push(IndexedChunk {
            chunk,
            embedding,
            index_id: id,
        });
        id
    }
    pub fn add_batch(&mut self, chunks: Vec<Chunk>, embeddings: Vec<Embedding>) -> Vec<usize> {
        chunks
            .into_iter()
            .zip(embeddings)
            .map(|(c, e)| self.add(c, e))
            .collect()
    }
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<SearchResult> {
        let mut scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, ic)| (i, query.cosine_similarity(&ic.embedding)))
            .collect();
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
            .into_iter()
            .map(|(i, score)| SearchResult {
                chunk: self.chunks[i].chunk.clone(),
                score,
                index_id: self.chunks[i].index_id,
            })
            .collect()
    }
    pub fn search_with_filter<F: Fn(&Chunk) -> bool>(
        &self,
        query: &Embedding,
        k: usize,
        filter: F,
    ) -> Vec<SearchResult> {
        let mut scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, ic)| filter(&ic.chunk))
            .map(|(i, ic)| (i, query.cosine_similarity(&ic.embedding)))
            .collect();
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
            .into_iter()
            .map(|(i, score)| SearchResult {
                chunk: self.chunks[i].chunk.clone(),
                score,
                index_id: self.chunks[i].index_id,
            })
            .collect()
    }
    pub fn remove_document(&mut self, doc_id: &str) -> usize {
        if let Some(indices) = self.doc_chunks.remove(doc_id) {
            let c = indices.len();
            self.chunks.retain(|c| c.chunk.doc_id != doc_id);
            c
        } else {
            0
        }
    }
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
    pub fn document_count(&self) -> usize {
        self.doc_chunks.len()
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub index_id: usize,
}

#[derive(Debug, Clone)]
pub struct ContextBuilder {
    max_tokens: usize,
    template: String,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new(MAX_CONTEXT_TOKENS)
    }
}

impl ContextBuilder {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            template: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".into(),
        }
    }
    #[must_use]
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }

    pub fn build(&self, query: &str, results: &[SearchResult]) -> Result<String> {
        let mut parts = Vec::new();
        let mut tokens = query.split_whitespace().count() + 20;
        for r in results {
            let ct = r.chunk.token_count();
            if tokens + ct > self.max_tokens {
                break;
            }
            parts.push(r.chunk.content.clone());
            tokens += ct;
        }
        if parts.is_empty() && !results.is_empty() {
            return Err(RagError::ContextOverflow {
                max_tokens: self.max_tokens,
                required: results[0].chunk.token_count(),
            });
        }
        Ok(self
            .template
            .replace("{context}", &parts.join("\n\n"))
            .replace("{query}", query))
    }
}

#[derive(Debug)]
pub struct RagPipeline {
    chunker: Chunker,
    embedder: EmbeddingModel,
    index: VectorIndex,
    context_builder: ContextBuilder,
}

impl Default for RagPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl RagPipeline {
    pub fn new() -> Self {
        Self {
            chunker: Chunker::default(),
            embedder: EmbeddingModel::default(),
            index: VectorIndex::new("default"),
            context_builder: ContextBuilder::default(),
        }
    }
    pub fn with_components(
        chunker: Chunker,
        embedder: EmbeddingModel,
        context_builder: ContextBuilder,
    ) -> Self {
        Self {
            chunker,
            embedder,
            index: VectorIndex::new("default"),
            context_builder,
        }
    }
    pub fn ingest(&mut self, doc: &Document) -> Result<usize> {
        let chunks = self.chunker.chunk(doc)?;
        let n = chunks.len();
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embedder.embed_batch(&texts)?;
        self.index.add_batch(chunks, embeddings);
        Ok(n)
    }
    pub fn ingest_batch(&mut self, docs: &[Document]) -> Result<usize> {
        docs.iter().try_fold(0, |acc, d| Ok(acc + self.ingest(d)?))
    }
    pub fn query(&self, query: &str, k: usize) -> Result<QueryResult> {
        let qe = self.embedder.embed(query)?;
        let sr = self.index.search(&qe, k);
        let ctx = self.context_builder.build(query, &sr)?;
        Ok(QueryResult {
            query: query.to_string(),
            results: sr,
            context: ctx,
        })
    }
    pub fn query_with_filter<F: Fn(&Chunk) -> bool>(
        &self,
        query: &str,
        k: usize,
        filter: F,
    ) -> Result<QueryResult> {
        let qe = self.embedder.embed(query)?;
        let sr = self.index.search_with_filter(&qe, k, filter);
        let ctx = self.context_builder.build(query, &sr)?;
        Ok(QueryResult {
            query: query.to_string(),
            results: sr,
            context: ctx,
        })
    }
    pub fn remove_document(&mut self, doc_id: &str) -> usize {
        self.index.remove_document(doc_id)
    }
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            chunk_count: self.index.len(),
            document_count: self.index.document_count(),
            embedding_dim: EMBEDDING_DIM,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub context: String,
}

impl QueryResult {
    pub fn top_result(&self) -> Option<&SearchResult> {
        self.results.first()
    }
    pub fn avg_score(&self) -> f32 {
        if self.results.is_empty() {
            0.0
        } else {
            self.results.iter().map(|r| r.score).sum::<f32>() / self.results.len() as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub chunk_count: usize,
    pub document_count: usize,
    pub embedding_dim: usize,
}

pub fn generate_sample_corpus(count: usize, _seed: u64) -> Vec<Document> {
    let topics = [
        "Machine learning is AI.",
        "Neural networks mimic neurons.",
        "Deep learning uses layers.",
        "NLP understands text.",
        "Vision interprets images.",
        "RL trains via rewards.",
        "Transfer learning reuses knowledge.",
        "Attention focuses on relevant info.",
        "Transformers model sequences.",
        "Embeddings are vector spaces.",
    ];
    (0..count)
        .map(|i| {
            let ti = i % topics.len();
            Document::new(
                format!("doc_{i}"),
                format!("{} Doc {i} topic {ti}.", topics[ti]),
            )
            .with_metadata("topic", format!("{ti}"))
            .with_metadata("index", format!("{i}"))
        })
        .collect()
}

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
