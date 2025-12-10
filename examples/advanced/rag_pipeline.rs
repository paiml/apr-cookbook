//! # Demo K: RAG Pipeline Integration
//!
//! Implements a complete Retrieval-Augmented Generation pipeline demonstrating
//! document chunking, embedding generation, vector similarity search, and
//! context injection for knowledge-grounded generation.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Quality built-in with 25-point checklist and property tests
//! - **Heijunka**: Level loading through efficient batched embedding
//! - **Genchi Genbutsu**: Go and see - comprehensive retrieval metrics
//! - **Kaizen**: Continuous improvement through index updates
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
//! │ Document │───▶│ Chunking │───▶│ Embedder │───▶│ VectorDB │
//! │ Source   │    │ Strategy │    │ (.apr)   │    │ (HNSW)   │
//! └──────────┘    └──────────┘    └──────────┘    └──────────┘
//!                                                      ▲
//!                                                      │ (Query)
//! ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
//! │ User     │───▶│ Prompt   │───▶│ Context  │◀───│ Retriever│
//! │ Query    │    │ Template │    │ Fuser    │    │ (Top-k)  │
//! └──────────┘    └──────────┘    └──────────┘    └──────────┘
//! ```
//!
//! ## 25-Point QA Checklist
//!
//! 1. Build succeeds
//! 2. Tests pass (100%)
//! 3. Clippy clean (0 warnings)
//! 4. Format clean
//! 5. Documentation >90%
//! 6. Unit test coverage >95%
//! 7. Property tests (100+ cases)
//! 8. No unwrap() in logic paths
//! 9. Proper error handling
//! 10. Deterministic index
//! 11. Retrieval precision >0.8
//! 12. Context window respected
//! 13. Query latency <20ms
//! 14. Index size <1.2x raw
//! 15. Handles duplicates
//! 16. Handles OOV words
//! 17. Batch query support
//! 18. Metadata filtering
//! 19. Index serialization
//! 20. Incremental updates
//! 21. IIUR compliance
//! 22. Toyota Way documented
//! 23. Example corpus included
//! 24. Hallucination check
//! 25. Memory cleanup
//!
//! ## Citations
//!
//! - Lewis et al. (2020) - RAG: Retrieval-Augmented Generation
//! - Karpukhin et al. (2020) - Dense Passage Retrieval
//! - Malkov & Yashunin (2018) - HNSW Algorithm

use std::collections::HashMap;

/// Default embedding dimension
pub const EMBEDDING_DIM: usize = 384;

/// Maximum context window size (tokens)
pub const MAX_CONTEXT_TOKENS: usize = 4096;

/// Error types for RAG operations
#[derive(Debug, Clone, PartialEq)]
pub enum RagError {
    /// Empty document
    EmptyDocument,
    /// Invalid embedding dimension
    InvalidEmbeddingDim { expected: usize, got: usize },
    /// Index not found
    IndexNotFound(String),
    /// Document not found
    DocumentNotFound(String),
    /// Chunking error
    ChunkingError(String),
    /// Embedding error
    EmbeddingError(String),
    /// Search error
    SearchError(String),
    /// Context overflow
    ContextOverflow { max_tokens: usize, required: usize },
}

impl std::fmt::Display for RagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDocument => write!(f, "Empty document"),
            Self::InvalidEmbeddingDim { expected, got } => {
                write!(f, "Invalid embedding dim: expected {expected}, got {got}")
            }
            Self::IndexNotFound(name) => write!(f, "Index not found: {name}"),
            Self::DocumentNotFound(id) => write!(f, "Document not found: {id}"),
            Self::ChunkingError(msg) => write!(f, "Chunking error: {msg}"),
            Self::EmbeddingError(msg) => write!(f, "Embedding error: {msg}"),
            Self::SearchError(msg) => write!(f, "Search error: {msg}"),
            Self::ContextOverflow {
                max_tokens,
                required,
            } => {
                write!(f, "Context overflow: max {max_tokens}, required {required}")
            }
        }
    }
}

impl std::error::Error for RagError {}

/// Result type for RAG operations
pub type Result<T> = std::result::Result<T, RagError>;

/// Document representation
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Estimate token count (simple whitespace-based)
    pub fn token_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// Document chunk with position info
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Parent document ID
    pub doc_id: String,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Chunk content
    pub content: String,
    /// Start character offset
    pub start_offset: usize,
    /// End character offset
    pub end_offset: usize,
}

impl Chunk {
    /// Estimate token count
    pub fn token_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    /// Fixed size chunks with overlap
    FixedSize { chunk_size: usize, overlap: usize },
    /// Sentence-based chunking
    Sentence { max_sentences: usize },
    /// Paragraph-based chunking
    Paragraph,
    /// Recursive token-based (like LangChain)
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

/// Document chunker
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
    /// Create a new chunker with given strategy
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Chunk a document
    pub fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        if doc.content.is_empty() {
            return Err(RagError::EmptyDocument);
        }

        match self.strategy {
            ChunkingStrategy::FixedSize {
                chunk_size,
                overlap,
            } => self.chunk_fixed_size(doc, chunk_size, overlap),
            ChunkingStrategy::Sentence { max_sentences } => {
                self.chunk_by_sentence(doc, max_sentences)
            }
            ChunkingStrategy::Paragraph => self.chunk_by_paragraph(doc),
            ChunkingStrategy::RecursiveToken {
                chunk_size,
                overlap,
            } => self.chunk_recursive_token(doc, chunk_size, overlap),
        }
    }

    /// Fixed size chunking
    fn chunk_fixed_size(
        &self,
        doc: &Document,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Vec<Chunk>> {
        let chars: Vec<char> = doc.content.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index,
                content,
                start_offset: start,
                end_offset: end,
            });

            chunk_index += 1;
            start = if end == chars.len() {
                chars.len()
            } else {
                end.saturating_sub(overlap)
            };
        }

        Ok(chunks)
    }

    /// Sentence-based chunking
    fn chunk_by_sentence(&self, doc: &Document, max_sentences: usize) -> Result<Vec<Chunk>> {
        let sentences: Vec<&str> = doc
            .content
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_offset = 0;

        for (chunk_index, sentence_group) in sentences.chunks(max_sentences).enumerate() {
            let content = sentence_group.join(". ").trim().to_string() + ".";
            let end_offset = current_offset + content.len();

            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index,
                content: content.clone(),
                start_offset: current_offset,
                end_offset,
            });

            current_offset = end_offset;
        }

        Ok(chunks)
    }

    /// Paragraph-based chunking
    fn chunk_by_paragraph(&self, doc: &Document) -> Result<Vec<Chunk>> {
        let paragraphs: Vec<&str> = doc
            .content
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_offset = 0;

        for (chunk_index, para) in paragraphs.iter().enumerate() {
            let content = para.trim().to_string();
            let end_offset = current_offset + content.len();

            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index,
                content,
                start_offset: current_offset,
                end_offset,
            });

            current_offset = end_offset + 2; // Account for \n\n
        }

        Ok(chunks)
    }

    /// Recursive token-based chunking
    fn chunk_recursive_token(
        &self,
        doc: &Document,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Vec<Chunk>> {
        let words: Vec<&str> = doc.content.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_index = 0;
        let mut char_offset = 0;

        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            let content = words[start..end].join(" ");
            let content_len = content.len();

            chunks.push(Chunk {
                doc_id: doc.id.clone(),
                chunk_index,
                content,
                start_offset: char_offset,
                end_offset: char_offset + content_len,
            });

            char_offset += content_len + 1;
            chunk_index += 1;
            start = if end == words.len() {
                words.len()
            } else {
                end.saturating_sub(overlap)
            };
        }

        Ok(chunks)
    }
}

/// Embedding vector
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Vector values
    pub values: Vec<f32>,
    /// Original text (for debugging)
    pub source_text: String,
}

impl Embedding {
    /// Create a new embedding
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

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// L2 distance
    pub fn l2_distance(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Simple embedding model (simulated)
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
    /// Create a new embedding model
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }

    /// Generate embedding for text (simulated with hash-based projection)
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        if text.is_empty() {
            return Err(RagError::EmbeddingError("Empty text".to_string()));
        }

        let mut values = vec![0.0_f32; self.dim];

        // Simple hash-based embedding (deterministic)
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let hash = self.hash_word(word, i as u64);
            for (j, v) in values.iter_mut().enumerate() {
                let angle = (hash.wrapping_add(j as u64) as f32) * 0.0001;
                *v += angle.sin() / words.len() as f32;
            }
        }

        // Normalize to unit vector
        let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut values {
                *v /= norm;
            }
        }

        Embedding::new(values, text.to_string())
    }

    /// Batch embedding
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Simple word hash
    fn hash_word(&self, word: &str, position: u64) -> u64 {
        let mut hash = self.seed.wrapping_add(position);
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }
}

/// Indexed chunk with embedding
#[derive(Debug, Clone)]
pub struct IndexedChunk {
    /// Chunk data
    pub chunk: Chunk,
    /// Embedding vector
    pub embedding: Embedding,
    /// Unique index ID
    pub index_id: usize,
}

/// Vector index using brute-force search (simplified HNSW)
#[derive(Debug)]
pub struct VectorIndex {
    /// Index name
    pub name: String,
    /// Indexed chunks
    chunks: Vec<IndexedChunk>,
    /// Document ID to chunk indices mapping
    doc_chunks: HashMap<String, Vec<usize>>,
    /// Next index ID
    next_id: usize,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            chunks: Vec::new(),
            doc_chunks: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a chunk to the index
    pub fn add(&mut self, chunk: Chunk, embedding: Embedding) -> usize {
        let index_id = self.next_id;
        self.next_id += 1;

        self.doc_chunks
            .entry(chunk.doc_id.clone())
            .or_default()
            .push(index_id);

        self.chunks.push(IndexedChunk {
            chunk,
            embedding,
            index_id,
        });

        index_id
    }

    /// Add multiple chunks
    pub fn add_batch(&mut self, chunks: Vec<Chunk>, embeddings: Vec<Embedding>) -> Vec<usize> {
        chunks
            .into_iter()
            .zip(embeddings)
            .map(|(c, e)| self.add(c, e))
            .collect()
    }

    /// Search for similar chunks
    pub fn search(&self, query_embedding: &Embedding, k: usize) -> Vec<SearchResult> {
        let mut scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, indexed)| (i, query_embedding.cosine_similarity(&indexed.embedding)))
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

    /// Search with metadata filter
    pub fn search_with_filter<F>(
        &self,
        query_embedding: &Embedding,
        k: usize,
        filter: F,
    ) -> Vec<SearchResult>
    where
        F: Fn(&Chunk) -> bool,
    {
        let mut scores: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, indexed)| filter(&indexed.chunk))
            .map(|(i, indexed)| (i, query_embedding.cosine_similarity(&indexed.embedding)))
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

    /// Remove document from index
    pub fn remove_document(&mut self, doc_id: &str) -> usize {
        if let Some(indices) = self.doc_chunks.remove(doc_id) {
            let count = indices.len();
            self.chunks.retain(|c| c.chunk.doc_id != doc_id);
            count
        } else {
            0
        }
    }

    /// Get index size
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get document count
    pub fn document_count(&self) -> usize {
        self.doc_chunks.len()
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Matched chunk
    pub chunk: Chunk,
    /// Similarity score
    pub score: f32,
    /// Index ID
    pub index_id: usize,
}

/// Context builder for prompt construction
#[derive(Debug, Clone)]
pub struct ContextBuilder {
    /// Maximum tokens in context
    max_tokens: usize,
    /// Context template
    template: String,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new(MAX_CONTEXT_TOKENS)
    }
}

impl ContextBuilder {
    /// Create a new context builder
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            template: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".to_string(),
        }
    }

    /// Set custom template
    #[must_use]
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }

    /// Build context from search results
    pub fn build(&self, query: &str, results: &[SearchResult]) -> Result<String> {
        let mut context_parts = Vec::new();
        let mut total_tokens = query.split_whitespace().count() + 20; // Template overhead

        for result in results {
            let chunk_tokens = result.chunk.token_count();
            if total_tokens + chunk_tokens > self.max_tokens {
                break;
            }
            context_parts.push(result.chunk.content.clone());
            total_tokens += chunk_tokens;
        }

        if context_parts.is_empty() && !results.is_empty() {
            return Err(RagError::ContextOverflow {
                max_tokens: self.max_tokens,
                required: results[0].chunk.token_count(),
            });
        }

        let context = context_parts.join("\n\n");
        Ok(self
            .template
            .replace("{context}", &context)
            .replace("{query}", query))
    }

    /// Build context respecting token limit
    pub fn build_with_limit(&self, query: &str, results: &[SearchResult], limit: usize) -> String {
        let effective_limit = limit.min(self.max_tokens);
        let mut context_parts = Vec::new();
        let mut total_tokens = query.split_whitespace().count() + 20;

        for result in results {
            let chunk_tokens = result.chunk.token_count();
            if total_tokens + chunk_tokens > effective_limit {
                break;
            }
            context_parts.push(result.chunk.content.clone());
            total_tokens += chunk_tokens;
        }

        let context = context_parts.join("\n\n");
        self.template
            .replace("{context}", &context)
            .replace("{query}", query)
    }
}

/// RAG pipeline orchestrator
#[derive(Debug)]
pub struct RagPipeline {
    /// Document chunker
    chunker: Chunker,
    /// Embedding model
    embedder: EmbeddingModel,
    /// Vector index
    index: VectorIndex,
    /// Context builder
    context_builder: ContextBuilder,
}

impl Default for RagPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl RagPipeline {
    /// Create a new RAG pipeline
    pub fn new() -> Self {
        Self {
            chunker: Chunker::default(),
            embedder: EmbeddingModel::default(),
            index: VectorIndex::new("default"),
            context_builder: ContextBuilder::default(),
        }
    }

    /// Create with custom components
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

    /// Ingest a document
    pub fn ingest(&mut self, doc: &Document) -> Result<usize> {
        let chunks = self.chunker.chunk(doc)?;
        let chunk_count = chunks.len();

        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.embedder.embed_batch(&texts)?;

        self.index.add_batch(chunks, embeddings);
        Ok(chunk_count)
    }

    /// Ingest multiple documents
    pub fn ingest_batch(&mut self, docs: &[Document]) -> Result<usize> {
        let mut total = 0;
        for doc in docs {
            total += self.ingest(doc)?;
        }
        Ok(total)
    }

    /// Query the pipeline
    pub fn query(&self, query: &str, k: usize) -> Result<QueryResult> {
        let query_embedding = self.embedder.embed(query)?;
        let search_results = self.index.search(&query_embedding, k);
        let context = self.context_builder.build(query, &search_results)?;

        Ok(QueryResult {
            query: query.to_string(),
            results: search_results,
            context,
        })
    }

    /// Query with metadata filter
    pub fn query_with_filter<F>(&self, query: &str, k: usize, filter: F) -> Result<QueryResult>
    where
        F: Fn(&Chunk) -> bool,
    {
        let query_embedding = self.embedder.embed(query)?;
        let search_results = self.index.search_with_filter(&query_embedding, k, filter);
        let context = self.context_builder.build(query, &search_results)?;

        Ok(QueryResult {
            query: query.to_string(),
            results: search_results,
            context,
        })
    }

    /// Remove a document
    pub fn remove_document(&mut self, doc_id: &str) -> usize {
        self.index.remove_document(doc_id)
    }

    /// Get index statistics
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            chunk_count: self.index.len(),
            document_count: self.index.document_count(),
            embedding_dim: EMBEDDING_DIM,
        }
    }
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Original query
    pub query: String,
    /// Search results
    pub results: Vec<SearchResult>,
    /// Built context
    pub context: String,
}

impl QueryResult {
    /// Get top result
    pub fn top_result(&self) -> Option<&SearchResult> {
        self.results.first()
    }

    /// Average score
    pub fn avg_score(&self) -> f32 {
        if self.results.is_empty() {
            0.0
        } else {
            self.results.iter().map(|r| r.score).sum::<f32>() / self.results.len() as f32
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total chunks indexed
    pub chunk_count: usize,
    /// Total documents
    pub document_count: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// Generate sample corpus for testing
pub fn generate_sample_corpus(count: usize, _seed: u64) -> Vec<Document> {
    let topics = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "Transfer learning reuses knowledge across tasks.",
        "Attention mechanisms focus on relevant information.",
        "Transformers revolutionized sequence modeling.",
        "Embeddings represent data in continuous vector spaces.",
    ];

    (0..count)
        .map(|i| {
            let topic_idx = i % topics.len();
            let content = format!(
                "{} This is document number {}. It contains information about topic {}.",
                topics[topic_idx], i, topic_idx
            );
            Document::new(format!("doc_{i}"), content)
                .with_metadata("topic", format!("{topic_idx}"))
                .with_metadata("index", format!("{i}"))
        })
        .collect()
}

fn main() {
    println!("=== Demo K: RAG Pipeline Integration ===\n");

    // Create pipeline
    let mut pipeline = RagPipeline::new();

    // Generate and ingest sample corpus
    let corpus = generate_sample_corpus(20, 42);
    println!("Generated {} documents", corpus.len());

    let chunk_count = pipeline
        .ingest_batch(&corpus)
        .expect("Failed to ingest corpus");
    println!("Indexed {} chunks", chunk_count);

    // Show stats
    let stats = pipeline.stats();
    println!(
        "\nPipeline stats: {} chunks, {} documents, {} dimensions",
        stats.chunk_count, stats.document_count, stats.embedding_dim
    );

    // Query examples
    let queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain transformers",
    ];

    println!("\n--- Query Results ---\n");
    for query in &queries {
        let result = pipeline.query(query, 3).expect("Failed to query");

        println!("Query: {}", query);
        println!("Top results:");
        for (i, r) in result.results.iter().take(3).enumerate() {
            println!(
                "  {}. [score: {:.3}] {}...",
                i + 1,
                r.score,
                &r.chunk.content[..r.chunk.content.len().min(50)]
            );
        }
        println!("Avg score: {:.3}", result.avg_score());
        println!();
    }

    // Chunking strategies demo
    println!("--- Chunking Strategies ---\n");

    let long_doc = Document::new(
        "long_doc",
        "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
    );

    let strategies = [
        (
            "Fixed(20, 5)",
            ChunkingStrategy::FixedSize {
                chunk_size: 20,
                overlap: 5,
            },
        ),
        (
            "Sentence(2)",
            ChunkingStrategy::Sentence { max_sentences: 2 },
        ),
        ("Paragraph", ChunkingStrategy::Paragraph),
        (
            "Recursive(10, 2)",
            ChunkingStrategy::RecursiveToken {
                chunk_size: 10,
                overlap: 2,
            },
        ),
    ];

    for (name, strategy) in strategies {
        let chunker = Chunker::new(strategy);
        let chunks = chunker.chunk(&long_doc).expect("Failed to chunk");
        println!("{}: {} chunks", name, chunks.len());
    }

    // Embedding similarity demo
    println!("\n--- Embedding Similarity ---\n");
    let embedder = EmbeddingModel::default();

    let text1 = "Machine learning is powerful";
    let text2 = "ML is a strong technology";
    let text3 = "The weather is nice today";

    let emb1 = embedder.embed(text1).expect("Failed to embed");
    let emb2 = embedder.embed(text2).expect("Failed to embed");
    let emb3 = embedder.embed(text3).expect("Failed to embed");

    println!(
        "Similarity('{}', '{}') = {:.3}",
        text1,
        text2,
        emb1.cosine_similarity(&emb2)
    );
    println!(
        "Similarity('{}', '{}') = {:.3}",
        text1,
        text3,
        emb1.cosine_similarity(&emb3)
    );

    println!("\n=== Demo Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("test", "Hello world");
        assert_eq!(doc.id, "test");
        assert_eq!(doc.content, "Hello world");
        assert_eq!(doc.token_count(), 2);
    }

    #[test]
    fn test_document_metadata() {
        let doc = Document::new("test", "Content").with_metadata("key", "value");
        assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_chunker_fixed_size() {
        let doc = Document::new("test", "ABCDEFGHIJ");
        let chunker = Chunker::new(ChunkingStrategy::FixedSize {
            chunk_size: 4,
            overlap: 1,
        });

        let chunks = chunker.chunk(&doc).expect("should work");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].content, "ABCD");
    }

    #[test]
    fn test_chunker_sentence() {
        let doc = Document::new("test", "First. Second. Third.");
        let chunker = Chunker::new(ChunkingStrategy::Sentence { max_sentences: 2 });

        let chunks = chunker.chunk(&doc).expect("should work");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunker_paragraph() {
        let doc = Document::new("test", "Para 1\n\nPara 2\n\nPara 3");
        let chunker = Chunker::new(ChunkingStrategy::Paragraph);

        let chunks = chunker.chunk(&doc).expect("should work");
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunker_empty_doc() {
        let doc = Document::new("test", "");
        let chunker = Chunker::default();

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_creation() {
        let values = vec![0.1_f32; EMBEDDING_DIM];
        let emb = Embedding::new(values, "test".to_string());
        assert!(emb.is_ok());
    }

    #[test]
    fn test_embedding_wrong_dim() {
        let values = vec![0.1_f32; 10];
        let emb = Embedding::new(values, "test".to_string());
        assert!(emb.is_err());
    }

    #[test]
    fn test_embedding_cosine_similarity() {
        let _emb1 = Embedding::new(vec![1.0, 0.0, 0.0], "a".to_string());
        let _emb2 = Embedding::new(vec![1.0, 0.0, 0.0], "b".to_string());
        // These will fail due to dim mismatch, using full dim
        let mut v1 = vec![0.0_f32; EMBEDDING_DIM];
        let mut v2 = vec![0.0_f32; EMBEDDING_DIM];
        v1[0] = 1.0;
        v2[0] = 1.0;

        let e1 = Embedding::new(v1, "a".to_string()).expect("should work");
        let e2 = Embedding::new(v2, "b".to_string()).expect("should work");

        let sim = e1.cosine_similarity(&e2);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_orthogonal() {
        let mut v1 = vec![0.0_f32; EMBEDDING_DIM];
        let mut v2 = vec![0.0_f32; EMBEDDING_DIM];
        v1[0] = 1.0;
        v2[1] = 1.0;

        let e1 = Embedding::new(v1, "a".to_string()).expect("should work");
        let e2 = Embedding::new(v2, "b".to_string()).expect("should work");

        let sim = e1.cosine_similarity(&e2);
        assert!(sim.abs() < 0.01);
    }

    #[test]
    fn test_embedding_model() {
        let model = EmbeddingModel::default();
        let emb = model.embed("Hello world");
        assert!(emb.is_ok());

        let e = emb.expect("should work");
        assert_eq!(e.values.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embedding_model_deterministic() {
        let model = EmbeddingModel::new(EMBEDDING_DIM, 42);
        let e1 = model.embed("Test text").expect("should work");
        let e2 = model.embed("Test text").expect("should work");

        assert_eq!(e1.values, e2.values);
    }

    #[test]
    fn test_embedding_model_empty() {
        let model = EmbeddingModel::default();
        let result = model.embed("");
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_add() {
        let mut index = VectorIndex::new("test");
        let chunk = Chunk {
            doc_id: "doc1".to_string(),
            chunk_index: 0,
            content: "Test content".to_string(),
            start_offset: 0,
            end_offset: 12,
        };
        let emb =
            Embedding::new(vec![0.1_f32; EMBEDDING_DIM], "test".to_string()).expect("should work");

        let id = index.add(chunk, emb);
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_index_search() {
        let mut index = VectorIndex::new("test");
        let model = EmbeddingModel::default();

        // Add some chunks
        for i in 0..5 {
            let chunk = Chunk {
                doc_id: format!("doc{i}"),
                chunk_index: 0,
                content: format!("Content {i}"),
                start_offset: 0,
                end_offset: 10,
            };
            let emb = model.embed(&chunk.content).expect("should work");
            index.add(chunk, emb);
        }

        let query_emb = model.embed("Content 0").expect("should work");
        let results = index.search(&query_emb, 3);

        assert_eq!(results.len(), 3);
        // First result should have highest score
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_vector_index_remove() {
        let mut index = VectorIndex::new("test");
        let model = EmbeddingModel::default();

        for i in 0..3 {
            let chunk = Chunk {
                doc_id: "doc1".to_string(),
                chunk_index: i,
                content: format!("Chunk {i}"),
                start_offset: 0,
                end_offset: 7,
            };
            let emb = model.embed(&chunk.content).expect("should work");
            index.add(chunk, emb);
        }

        assert_eq!(index.len(), 3);
        let removed = index.remove_document("doc1");
        assert_eq!(removed, 3);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_context_builder() {
        let builder = ContextBuilder::default();
        let results = vec![SearchResult {
            chunk: Chunk {
                doc_id: "doc1".to_string(),
                chunk_index: 0,
                content: "Relevant information".to_string(),
                start_offset: 0,
                end_offset: 20,
            },
            score: 0.9,
            index_id: 0,
        }];

        let context = builder.build("What is this?", &results);
        assert!(context.is_ok());

        let ctx = context.expect("should work");
        assert!(ctx.contains("Relevant information"));
        assert!(ctx.contains("What is this?"));
    }

    #[test]
    fn test_rag_pipeline_ingest() {
        let mut pipeline = RagPipeline::new();
        let doc = Document::new("test", "This is test content for the pipeline.");

        let count = pipeline.ingest(&doc);
        assert!(count.is_ok());
        assert!(count.expect("should work") > 0);
    }

    #[test]
    fn test_rag_pipeline_query() {
        let mut pipeline = RagPipeline::new();
        let docs = generate_sample_corpus(5, 42);

        pipeline.ingest_batch(&docs).expect("should work");

        let result = pipeline.query("machine learning", 3);
        assert!(result.is_ok());

        let r = result.expect("should work");
        assert!(!r.results.is_empty());
    }

    #[test]
    fn test_rag_pipeline_stats() {
        let mut pipeline = RagPipeline::new();
        let docs = generate_sample_corpus(3, 42);

        pipeline.ingest_batch(&docs).expect("should work");

        let stats = pipeline.stats();
        assert!(stats.chunk_count > 0);
        assert_eq!(stats.document_count, 3);
    }

    #[test]
    fn test_query_result_avg_score() {
        let result = QueryResult {
            query: "test".to_string(),
            results: vec![
                SearchResult {
                    chunk: Chunk {
                        doc_id: "d1".to_string(),
                        chunk_index: 0,
                        content: "c1".to_string(),
                        start_offset: 0,
                        end_offset: 2,
                    },
                    score: 0.8,
                    index_id: 0,
                },
                SearchResult {
                    chunk: Chunk {
                        doc_id: "d2".to_string(),
                        chunk_index: 0,
                        content: "c2".to_string(),
                        start_offset: 0,
                        end_offset: 2,
                    },
                    score: 0.6,
                    index_id: 1,
                },
            ],
            context: "test".to_string(),
        };

        assert!((result.avg_score() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_generate_sample_corpus() {
        let corpus = generate_sample_corpus(10, 42);
        assert_eq!(corpus.len(), 10);

        for (i, doc) in corpus.iter().enumerate() {
            assert_eq!(doc.id, format!("doc_{i}"));
            assert!(!doc.content.is_empty());
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_document_token_count(content in "[a-z ]{1,100}") {
            let doc = Document::new("test", &content);
            let count = doc.token_count();
            prop_assert!(count <= content.len());
        }

        #[test]
        fn prop_chunker_non_empty(content in "[a-z]{10,100}") {
            let doc = Document::new("test", &content);
            let chunker = Chunker::default();
            let chunks = chunker.chunk(&doc);
            prop_assert!(chunks.is_ok());
            prop_assert!(!chunks.expect("should work").is_empty());
        }

        #[test]
        fn prop_embedding_normalized(text in "[a-z ]{5,50}") {
            let model = EmbeddingModel::default();
            let emb = model.embed(&text);

            if let Ok(e) = emb {
                let norm: f32 = e.values.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 0.01);
            }
        }

        #[test]
        fn prop_cosine_similarity_bounds(
            seed1 in 0u64..1000,
            _seed2 in 0u64..1000
        ) {
            let model = EmbeddingModel::new(EMBEDDING_DIM, seed1);
            let e1 = model.embed("text one").expect("should work");
            let e2 = model.embed("text two").expect("should work");

            let sim = e1.cosine_similarity(&e2);
            prop_assert!(sim >= -1.0);
            prop_assert!(sim <= 1.0);
        }

        #[test]
        fn prop_self_similarity_is_one(text in "[a-z ]{5,30}") {
            let model = EmbeddingModel::default();
            if let Ok(emb) = model.embed(&text) {
                let sim = emb.cosine_similarity(&emb);
                prop_assert!((sim - 1.0).abs() < 0.01);
            }
        }

        #[test]
        fn prop_search_returns_at_most_k(k in 1usize..10) {
            let mut index = VectorIndex::new("test");
            let model = EmbeddingModel::default();

            for i in 0..5 {
                let chunk = Chunk {
                    doc_id: format!("doc{i}"),
                    chunk_index: 0,
                    content: format!("Content {i}"),
                    start_offset: 0,
                    end_offset: 10,
                };
                let emb = model.embed(&chunk.content).expect("should work");
                index.add(chunk, emb);
            }

            let query_emb = model.embed("query").expect("should work");
            let results = index.search(&query_emb, k);

            prop_assert!(results.len() <= k);
        }

        #[test]
        fn prop_search_results_sorted(k in 3usize..10) {
            let mut index = VectorIndex::new("test");
            let model = EmbeddingModel::default();

            for i in 0..10 {
                let chunk = Chunk {
                    doc_id: format!("doc{i}"),
                    chunk_index: 0,
                    content: format!("Content number {i}"),
                    start_offset: 0,
                    end_offset: 15,
                };
                let emb = model.embed(&chunk.content).expect("should work");
                index.add(chunk, emb);
            }

            let query_emb = model.embed("query text").expect("should work");
            let results = index.search(&query_emb, k);

            for i in 1..results.len() {
                prop_assert!(results[i-1].score >= results[i].score);
            }
        }

        #[test]
        fn prop_pipeline_ingest_count(doc_count in 1usize..10) {
            let mut pipeline = RagPipeline::new();
            let docs = generate_sample_corpus(doc_count, 42);

            let result = pipeline.ingest_batch(&docs);
            prop_assert!(result.is_ok());

            let stats = pipeline.stats();
            prop_assert_eq!(stats.document_count, doc_count);
        }

        #[test]
        fn prop_chunk_offsets_valid(content in "[a-z ]{20,100}") {
            let doc = Document::new("test", &content);
            let chunker = Chunker::new(ChunkingStrategy::FixedSize {
                chunk_size: 10,
                overlap: 2,
            });

            if let Ok(chunks) = chunker.chunk(&doc) {
                for chunk in &chunks {
                    prop_assert!(chunk.start_offset <= chunk.end_offset);
                    prop_assert!(chunk.end_offset <= content.len() + 10);
                }
            }
        }
    }
}
