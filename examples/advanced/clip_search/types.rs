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

pub const EMBEDDING_DIM: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub modality: Modality,
    pub normalized: bool,
}

impl Embedding {
    pub fn new(vector: Vec<f32>, modality: Modality) -> Self {
        let mut e = Self {
            vector,
            modality,
            normalized: false,
        };
        e.normalize();
        e
    }
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut self.vector {
                *v /= norm;
            }
            self.normalized = true;
        }
    }
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

#[derive(Debug, Clone)]
pub struct TextDocument {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}
impl TextDocument {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.insert(k.into(), v.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct ImageDocument {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub metadata: HashMap<String, String>,
}
impl ImageDocument {
    pub fn new(id: impl Into<String>, w: u32, h: u32, px: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            width: w,
            height: h,
            pixels: px,
            metadata: HashMap::new(),
        }
    }
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.insert(k.into(), v.into());
        self
    }
    pub fn test_pattern(id: impl Into<String>, w: u32, h: u32, seed: u32) -> Self {
        let px: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    [
                        ((x.wrapping_mul(seed) + y) % 256) as u8,
                        ((y.wrapping_mul(seed) + x) % 256) as u8,
                        ((x.wrapping_add(y).wrapping_mul(seed)) % 256) as u8,
                    ]
                })
            })
            .collect();
        Self::new(id, w, h, px)
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub modality: Modality,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

pub fn seeded_weights(rows: usize, cols: usize, state: &mut u64) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((*state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                })
                .collect()
        })
        .collect()
}

pub fn linear(input: &[f32], weights: &[Vec<f32>], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for (i, &v) in input.iter().enumerate() {
        if i < weights.len() {
            for (j, &w) in weights[i].iter().enumerate() {
                out[j] += v * w;
            }
        }
    }
    out
}

#[derive(Debug)]
pub struct TextEncoder {
    pub vocab: HashMap<String, usize>,
    pub emb_w: Vec<Vec<f32>>,
    pub proj_w: Vec<Vec<f32>>,
}

impl TextEncoder {
    pub fn new(vocab_size: usize, hidden: usize, seed: u64) -> Self {
        let mut s = seed;
        Self {
            vocab: HashMap::new(),
            emb_w: seeded_weights(vocab_size, hidden, &mut s),
            proj_w: seeded_weights(hidden, EMBEDDING_DIM, &mut s),
        }
    }
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }
    pub fn vocab_idx(&mut self, token: &str) -> usize {
        let vs = self.emb_w.len();
        let n = self.vocab.len();
        *self.vocab.entry(token.to_string()).or_insert(n % vs)
    }
    pub fn encode(&mut self, text: &str) -> Embedding {
        let tokens = self.tokenize(text);
        if tokens.is_empty() {
            return Embedding::new(vec![0.0; EMBEDDING_DIM], Modality::Text);
        }
        let hdim = self.emb_w[0].len();
        let mut pooled = vec![0.0_f32; hdim];
        for t in &tokens {
            let idx = self.vocab_idx(t);
            for (i, &w) in self.emb_w[idx].iter().enumerate() {
                pooled[i] += w;
            }
        }
        let n = tokens.len() as f32;
        for p in &mut pooled {
            *p /= n;
        }
        Embedding::new(linear(&pooled, &self.proj_w, EMBEDDING_DIM), Modality::Text)
    }
}

pub const PATCH_SIZE: u32 = 16;

#[derive(Debug)]
pub struct ImageEncoder {
    pub patch_proj: Vec<Vec<f32>>,
    pub pos_emb: Vec<Vec<f32>>,
    pub final_proj: Vec<Vec<f32>>,
    pub hdim: usize,
}

impl ImageEncoder {
    pub fn new(hdim: usize, max_patches: usize, seed: u64) -> Self {
        let mut s = seed;
        let pd = (3 * PATCH_SIZE * PATCH_SIZE) as usize;
        Self {
            patch_proj: seeded_weights(pd, hdim, &mut s),
            pos_emb: seeded_weights(max_patches, hdim, &mut s),
            final_proj: seeded_weights(hdim, EMBEDDING_DIM, &mut s),
            hdim,
        }
    }
    pub fn extract_patches(&self, img: &ImageDocument) -> Vec<Vec<f32>> {
        let (npx, npy) = (
            (img.width / PATCH_SIZE).max(1),
            (img.height / PATCH_SIZE).max(1),
        );
        let psz = (3 * PATCH_SIZE * PATCH_SIZE) as usize;
        (0..npy)
            .flat_map(|py| {
                (0..npx).map(move |px| {
                    let mut patch = Vec::with_capacity(psz);
                    for y in 0..PATCH_SIZE {
                        for x in 0..PATCH_SIZE {
                            let (ix, iy) = (
                                (px * PATCH_SIZE + x).min(img.width - 1),
                                (py * PATCH_SIZE + y).min(img.height - 1),
                            );
                            let idx = ((iy * img.width + ix) * 3) as usize;
                            if idx + 2 < img.pixels.len() {
                                for c in 0..3 {
                                    patch.push(f32::from(img.pixels[idx + c]) / 255.0);
                                }
                            } else {
                                patch.extend_from_slice(&[0.0, 0.0, 0.0]);
                            }
                        }
                    }
                    patch.resize(psz, 0.0);
                    patch
                })
            })
            .collect()
    }
    pub fn encode(&self, img: &ImageDocument) -> Embedding {
        let patches = self.extract_patches(img);
        if patches.is_empty() {
            return Embedding::new(vec![0.0; EMBEDDING_DIM], Modality::Image);
        }
        let mut pooled = vec![0.0_f32; self.hdim];
        for (i, patch) in patches.iter().enumerate() {
            let mut h = linear(patch, &self.patch_proj, self.hdim);
            if i < self.pos_emb.len() {
                for (k, &pe) in self.pos_emb[i].iter().enumerate() {
                    h[k] += pe;
                }
            }
            for (k, &v) in h.iter().enumerate() {
                pooled[k] += v;
            }
        }
        let n = patches.len() as f32;
        for p in &mut pooled {
            *p /= n;
        }
        Embedding::new(
            linear(&pooled, &self.final_proj, EMBEDDING_DIM),
            Modality::Image,
        )
    }
}

#[derive(Debug, Clone)]
pub struct IndexedItem {
    pub id: String,
    pub modality: Modality,
    pub embedding: Embedding,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ClipIndex {
    pub items: Vec<IndexedItem>,
    pub text_enc: TextEncoder,
    pub image_enc: ImageEncoder,
}

impl ClipIndex {
    pub fn new(seed: u64) -> Self {
        Self {
            items: Vec::new(),
            text_enc: TextEncoder::new(10000, 256, seed),
            image_enc: ImageEncoder::new(256, 256, seed.wrapping_add(1)),
        }
    }
    pub fn index_text(&mut self, doc: TextDocument) {
        let e = self.text_enc.encode(&doc.content);
        self.items.push(IndexedItem {
            id: doc.id,
            modality: Modality::Text,
            embedding: e,
            metadata: doc.metadata,
        });
    }
    pub fn index_image(&mut self, doc: ImageDocument) {
        let e = self.image_enc.encode(&doc);
        self.items.push(IndexedItem {
            id: doc.id,
            modality: Modality::Image,
            embedding: e,
            metadata: doc.metadata,
        });
    }
    pub fn index_texts(&mut self, docs: &[TextDocument]) {
        for d in docs {
            self.index_text(d.clone());
        }
    }
    pub fn index_images(&mut self, docs: &[ImageDocument]) {
        for d in docs {
            self.index_image(d.clone());
        }
    }
    pub fn search_by_text(&mut self, q: &str, k: usize) -> Vec<SearchResult> {
        let e = self.text_enc.encode(q);
        self.search(&e, k, None)
    }
    pub fn search_by_image(&mut self, img: &ImageDocument, k: usize) -> Vec<SearchResult> {
        let e = self.image_enc.encode(img);
        self.search(&e, k, None)
    }
    pub fn search_by_text_filtered(&mut self, q: &str, k: usize, m: Modality) -> Vec<SearchResult> {
        let e = self.text_enc.encode(q);
        self.search(&e, k, Some(m))
    }
    pub fn search(&self, q: &Embedding, k: usize, mf: Option<Modality>) -> Vec<SearchResult> {
        let mut scores: Vec<(usize, f32)> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| mf.map_or(true, |m| it.modality == m))
            .map(|(i, it)| (i, q.similarity(&it.embedding)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .into_iter()
            .take(k)
            .map(|(i, s)| {
                let it = &self.items[i];
                SearchResult {
                    id: it.id.clone(),
                    modality: it.modality,
                    score: s,
                    metadata: it.metadata.clone(),
                }
            })
            .collect()
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn count_by_modality(&self, m: Modality) -> usize {
        self.items.iter().filter(|i| i.modality == m).count()
    }
}

pub fn contrastive_loss(text_embs: &[Embedding], img_embs: &[Embedding], temp: f32) -> f32 {
    if text_embs.is_empty() || img_embs.is_empty() {
        return 0.0;
    }
    let n = text_embs.len().min(img_embs.len());
    let mut loss = 0.0_f32;
    for i in 0..n {
        for (src, tgt) in [(&text_embs[i], img_embs), (&img_embs[i], text_embs)] {
            let logits: Vec<f32> = tgt.iter().map(|t| src.similarity(t) / temp).collect();
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            loss -= logits[i] - mx - logits.iter().map(|&l| (l - mx).exp()).sum::<f32>().ln();
        }
    }
    loss / (2.0 * n as f32)
}

pub fn recall_at_k(results: &[SearchResult], expected: &str, k: usize) -> f32 {
    if results.iter().take(k).any(|r| r.id == expected) {
        1.0
    } else {
        0.0
    }
}
pub fn mean_reciprocal_rank(results: &[SearchResult], expected: &str) -> f32 {
    results
        .iter()
        .enumerate()
        .find(|(_, r)| r.id == expected)
        .map_or(0.0, |(i, _)| 1.0 / (i + 1) as f32)
}
