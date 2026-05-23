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
#[allow(unused_imports)]
use super::types::*;

use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::HashMap;

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

// --- Chunker impl ---

impl Chunker {
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

// --- EmbeddingModel impl ---

impl EmbeddingModel {
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

// --- VectorIndex impl ---

impl VectorIndex {
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
}

// --- ContextBuilder impl ---

impl ContextBuilder {
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

// --- RagPipeline impl ---

impl RagPipeline {
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
}
