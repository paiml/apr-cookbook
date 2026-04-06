#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use proptest::prelude::*;
#[allow(unused_imports)]
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
    pub strategy: ChunkingStrategy,
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

    pub fn chunk_fixed(&self, doc: &Document, size: usize, overlap: usize) -> Result<Vec<Chunk>> {
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

    pub fn chunk_sentence(&self, doc: &Document, max: usize) -> Result<Vec<Chunk>> {
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

    pub fn chunk_paragraph(&self, doc: &Document) -> Result<Vec<Chunk>> {
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

    pub fn chunk_recursive(
        &self,
        doc: &Document,
        size: usize,
        overlap: usize,
    ) -> Result<Vec<Chunk>> {
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
    pub dim: usize,
    pub seed: u64,
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

    pub fn hash_word(&self, word: &str, position: u64) -> u64 {
        let mut hash = self.seed.wrapping_add(position);
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }
}

#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub chunk: Chunk,
    pub embedding: Embedding,
    pub index_id: usize,
}

#[derive(Debug)]
pub struct VectorIndex {
    pub name: String,
    pub chunks: Vec<IndexedChunk>,
    pub doc_chunks: HashMap<String, Vec<usize>>,
    pub next_id: usize,
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
    pub max_tokens: usize,
    pub template: String,
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
    pub chunker: Chunker,
    pub embedder: EmbeddingModel,
    pub index: VectorIndex,
    pub context_builder: ContextBuilder,
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
