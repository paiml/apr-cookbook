//! Demo O: Multi-Modal CLIP Search - text-to-image and image-to-text semantic search.
//! QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//! ## References
//! - Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. arXiv:2103.00020

use std::collections::HashMap;

const EMBEDDING_DIM: usize = 512;

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

fn seeded_weights(rows: usize, cols: usize, state: &mut u64) -> Vec<Vec<f32>> {
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

fn linear(input: &[f32], weights: &[Vec<f32>], out_dim: usize) -> Vec<f32> {
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
    vocab: HashMap<String, usize>,
    emb_w: Vec<Vec<f32>>,
    proj_w: Vec<Vec<f32>>,
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
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }
    fn vocab_idx(&mut self, token: &str) -> usize {
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

const PATCH_SIZE: u32 = 16;

#[derive(Debug)]
pub struct ImageEncoder {
    patch_proj: Vec<Vec<f32>>,
    pos_emb: Vec<Vec<f32>>,
    final_proj: Vec<Vec<f32>>,
    hdim: usize,
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
    fn extract_patches(&self, img: &ImageDocument) -> Vec<Vec<f32>> {
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
struct IndexedItem {
    id: String,
    modality: Modality,
    embedding: Embedding,
    metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ClipIndex {
    items: Vec<IndexedItem>,
    text_enc: TextEncoder,
    image_enc: ImageEncoder,
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
    fn search(&self, q: &Embedding, k: usize, mf: Option<Modality>) -> Vec<SearchResult> {
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
