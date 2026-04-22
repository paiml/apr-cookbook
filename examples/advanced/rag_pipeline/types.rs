//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
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
