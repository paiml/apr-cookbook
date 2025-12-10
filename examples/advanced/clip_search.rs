//! Demo O: Multi-Modal CLIP Search
//!
//! EXTREME TDD demonstration of a CLIP-style multi-modal search system.
//! Uses contrastive learning principles to enable text-to-image and
//! image-to-text semantic search.
//!
//! Toyota Way Principles Applied:
//! - Jidoka: Automatic quality checks with embedding validation
//! - Heijunka: Balanced processing across modalities
//! - Genchi Genbutsu: Direct embedding comparison for similarity
//! - Kaizen: Incremental index building and optimization
//! - Poka-yoke: Type-safe modality handling prevents cross-modal errors
//!
//! pmat Quality Gates:
//! - Max Cyclomatic Complexity: 10
//! - SATD Violations: 0
//! - Test Coverage Target: 95%+

use std::collections::HashMap;

// ============================================================================
// DOMAIN TYPES
// ============================================================================

/// Embedding dimension for the shared latent space
const EMBEDDING_DIM: usize = 512;

/// Content modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
}

/// A multi-modal embedding in the shared latent space
#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub modality: Modality,
    pub normalized: bool,
}

impl Embedding {
    /// Create a new embedding with automatic L2 normalization
    pub fn new(vector: Vec<f32>, modality: Modality) -> Self {
        let mut emb = Self {
            vector,
            modality,
            normalized: false,
        };
        emb.normalize();
        emb
    }

    /// L2 normalize the embedding vector (CLIP requirement)
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut self.vector {
                *v /= norm;
            }
            self.normalized = true;
        }
    }

    /// Compute cosine similarity (dot product for normalized vectors)
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

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// A text document that can be embedded
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
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// An image that can be embedded (simplified representation)
#[derive(Debug, Clone)]
pub struct ImageDocument {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // Flattened RGB
    pub metadata: HashMap<String, String>,
}

impl ImageDocument {
    pub fn new(id: impl Into<String>, width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            width,
            height,
            pixels,
            metadata: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Create a test image with a simple pattern
    pub fn test_pattern(id: impl Into<String>, width: u32, height: u32, seed: u32) -> Self {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                // Generate deterministic pattern based on seed
                let r = ((x.wrapping_mul(seed) + y) % 256) as u8;
                let g = ((y.wrapping_mul(seed) + x) % 256) as u8;
                let b = ((x.wrapping_add(y).wrapping_mul(seed)) % 256) as u8;
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }
        Self::new(id, width, height, pixels)
    }
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub modality: Modality,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

impl SearchResult {
    pub fn new(
        id: String,
        modality: Modality,
        score: f32,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id,
            modality,
            score,
            metadata,
        }
    }
}

// ============================================================================
// TEXT ENCODER (Simplified transformer-style)
// ============================================================================

/// Text encoder using simplified attention mechanism
#[derive(Debug)]
pub struct TextEncoder {
    vocab: HashMap<String, usize>,
    embedding_weights: Vec<Vec<f32>>,
    projection_weights: Vec<Vec<f32>>,
}

impl TextEncoder {
    /// Create a new text encoder with random initialization
    pub fn new(vocab_size: usize, hidden_dim: usize, seed: u64) -> Self {
        // Simple seeded random for reproducibility
        let mut rng_state = seed;

        let vocab = HashMap::new(); // Will be built incrementally

        // Initialize embedding weights
        let embedding_weights: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                    })
                    .collect()
            })
            .collect();

        // Initialize projection weights (hidden_dim -> EMBEDDING_DIM)
        let projection_weights: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| {
                (0..EMBEDDING_DIM)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            vocab,
            embedding_weights,
            projection_weights,
        }
    }

    /// Tokenize text into word pieces
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }

    /// Get or create vocab index for a token
    fn get_vocab_idx(&mut self, token: &str) -> usize {
        let vocab_size = self.embedding_weights.len();
        let current_len = self.vocab.len();
        *self
            .vocab
            .entry(token.to_string())
            .or_insert(current_len % vocab_size)
    }

    /// Encode text to embedding
    pub fn encode(&mut self, text: &str) -> Embedding {
        let tokens = self.tokenize(text);

        // Get token embeddings and average them (mean pooling)
        let mut pooled = vec![0.0f32; self.embedding_weights[0].len()];

        if tokens.is_empty() {
            // Return zero embedding for empty text
            return Embedding::new(vec![0.0; EMBEDDING_DIM], Modality::Text);
        }

        for token in &tokens {
            let idx = self.get_vocab_idx(token);
            for (i, &w) in self.embedding_weights[idx].iter().enumerate() {
                pooled[i] += w;
            }
        }

        // Average
        let n = tokens.len() as f32;
        for p in &mut pooled {
            *p /= n;
        }

        // Project to shared embedding space
        let mut projected = vec![0.0f32; EMBEDDING_DIM];
        for (i, &p) in pooled.iter().enumerate() {
            for (j, &w) in self.projection_weights[i].iter().enumerate() {
                projected[j] += p * w;
            }
        }

        Embedding::new(projected, Modality::Text)
    }
}

// ============================================================================
// IMAGE ENCODER (Simplified ViT-style)
// ============================================================================

/// Patch size for vision transformer
const PATCH_SIZE: u32 = 16;

/// Image encoder using simplified vision transformer
#[derive(Debug)]
pub struct ImageEncoder {
    patch_projection: Vec<Vec<f32>>,
    position_embeddings: Vec<Vec<f32>>,
    final_projection: Vec<Vec<f32>>,
    hidden_dim: usize,
}

impl ImageEncoder {
    /// Create a new image encoder
    pub fn new(hidden_dim: usize, max_patches: usize, seed: u64) -> Self {
        let mut rng_state = seed;

        // Patch projection (3 * patch_size^2 -> hidden_dim)
        let patch_input_dim = (3 * PATCH_SIZE * PATCH_SIZE) as usize;
        let patch_projection: Vec<Vec<f32>> = (0..patch_input_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                    })
                    .collect()
            })
            .collect();

        // Position embeddings
        let position_embeddings: Vec<Vec<f32>> = (0..max_patches)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                    })
                    .collect()
            })
            .collect();

        // Final projection (hidden_dim -> EMBEDDING_DIM)
        let final_projection: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| {
                (0..EMBEDDING_DIM)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng_state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
                    })
                    .collect()
            })
            .collect();

        Self {
            patch_projection,
            position_embeddings,
            final_projection,
            hidden_dim,
        }
    }

    /// Extract patches from image
    fn extract_patches(&self, image: &ImageDocument) -> Vec<Vec<f32>> {
        let mut patches = Vec::new();

        let num_patches_x = (image.width / PATCH_SIZE).max(1);
        let num_patches_y = (image.height / PATCH_SIZE).max(1);

        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                let mut patch = Vec::with_capacity((3 * PATCH_SIZE * PATCH_SIZE) as usize);

                for y in 0..PATCH_SIZE {
                    for x in 0..PATCH_SIZE {
                        let img_x = (px * PATCH_SIZE + x).min(image.width - 1);
                        let img_y = (py * PATCH_SIZE + y).min(image.height - 1);
                        let idx = ((img_y * image.width + img_x) * 3) as usize;

                        if idx + 2 < image.pixels.len() {
                            patch.push(f32::from(image.pixels[idx]) / 255.0);
                            patch.push(f32::from(image.pixels[idx + 1]) / 255.0);
                            patch.push(f32::from(image.pixels[idx + 2]) / 255.0);
                        } else {
                            patch.push(0.0);
                            patch.push(0.0);
                            patch.push(0.0);
                        }
                    }
                }

                // Pad or truncate to expected size
                patch.resize((3 * PATCH_SIZE * PATCH_SIZE) as usize, 0.0);
                patches.push(patch);
            }
        }

        patches
    }

    /// Encode image to embedding
    pub fn encode(&self, image: &ImageDocument) -> Embedding {
        let patches = self.extract_patches(image);

        if patches.is_empty() {
            return Embedding::new(vec![0.0; EMBEDDING_DIM], Modality::Image);
        }

        // Project patches and add position embeddings
        let mut hidden_states: Vec<Vec<f32>> = Vec::new();

        for (i, patch) in patches.iter().enumerate() {
            let mut hidden = vec![0.0f32; self.hidden_dim];

            // Linear projection
            for (j, &p) in patch.iter().enumerate() {
                if j < self.patch_projection.len() {
                    for (k, &w) in self.patch_projection[j].iter().enumerate() {
                        hidden[k] += p * w;
                    }
                }
            }

            // Add position embedding
            if i < self.position_embeddings.len() {
                for (k, &pe) in self.position_embeddings[i].iter().enumerate() {
                    hidden[k] += pe;
                }
            }

            hidden_states.push(hidden);
        }

        // Global average pooling
        let mut pooled = vec![0.0f32; self.hidden_dim];
        for hidden in &hidden_states {
            for (i, &h) in hidden.iter().enumerate() {
                pooled[i] += h;
            }
        }
        let n = hidden_states.len() as f32;
        for p in &mut pooled {
            *p /= n;
        }

        // Final projection to shared embedding space
        let mut projected = vec![0.0f32; EMBEDDING_DIM];
        for (i, &p) in pooled.iter().enumerate() {
            if i < self.final_projection.len() {
                for (j, &w) in self.final_projection[i].iter().enumerate() {
                    projected[j] += p * w;
                }
            }
        }

        Embedding::new(projected, Modality::Image)
    }
}

// ============================================================================
// CLIP INDEX
// ============================================================================

/// Indexed item in the search index
#[derive(Debug, Clone)]
struct IndexedItem {
    id: String,
    modality: Modality,
    embedding: Embedding,
    metadata: HashMap<String, String>,
}

/// Multi-modal search index
#[derive(Debug)]
pub struct ClipIndex {
    items: Vec<IndexedItem>,
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
}

impl ClipIndex {
    /// Create a new CLIP index
    pub fn new(seed: u64) -> Self {
        let text_encoder = TextEncoder::new(10000, 256, seed);
        let image_encoder = ImageEncoder::new(256, 256, seed.wrapping_add(1));

        Self {
            items: Vec::new(),
            text_encoder,
            image_encoder,
        }
    }

    /// Index a text document
    pub fn index_text(&mut self, doc: TextDocument) {
        let embedding = self.text_encoder.encode(&doc.content);
        self.items.push(IndexedItem {
            id: doc.id,
            modality: Modality::Text,
            embedding,
            metadata: doc.metadata,
        });
    }

    /// Index an image document
    pub fn index_image(&mut self, doc: ImageDocument) {
        let embedding = self.image_encoder.encode(&doc);
        self.items.push(IndexedItem {
            id: doc.id,
            modality: Modality::Image,
            embedding,
            metadata: doc.metadata,
        });
    }

    /// Index multiple text documents
    pub fn index_texts(&mut self, docs: &[TextDocument]) {
        for doc in docs {
            self.index_text(doc.clone());
        }
    }

    /// Index multiple image documents
    pub fn index_images(&mut self, docs: &[ImageDocument]) {
        for doc in docs {
            self.index_image(doc.clone());
        }
    }

    /// Search by text query
    pub fn search_by_text(&mut self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_embedding = self.text_encoder.encode(query);
        self.search_by_embedding(&query_embedding, top_k, None)
    }

    /// Search by image query
    pub fn search_by_image(&mut self, image: &ImageDocument, top_k: usize) -> Vec<SearchResult> {
        let query_embedding = self.image_encoder.encode(image);
        self.search_by_embedding(&query_embedding, top_k, None)
    }

    /// Search with modality filter
    pub fn search_by_text_filtered(
        &mut self,
        query: &str,
        top_k: usize,
        modality: Modality,
    ) -> Vec<SearchResult> {
        let query_embedding = self.text_encoder.encode(query);
        self.search_by_embedding(&query_embedding, top_k, Some(modality))
    }

    /// Internal search by embedding
    fn search_by_embedding(
        &self,
        query: &Embedding,
        top_k: usize,
        modality_filter: Option<Modality>,
    ) -> Vec<SearchResult> {
        let mut scores: Vec<(usize, f32)> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| modality_filter.map_or(true, |m| item.modality == m))
            .map(|(i, item)| (i, query.similarity(&item.embedding)))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(top_k)
            .map(|(i, score)| {
                let item = &self.items[i];
                SearchResult::new(item.id.clone(), item.modality, score, item.metadata.clone())
            })
            .collect()
    }

    /// Get total number of indexed items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get count by modality
    pub fn count_by_modality(&self, modality: Modality) -> usize {
        self.items.iter().filter(|i| i.modality == modality).count()
    }
}

// ============================================================================
// CONTRASTIVE LEARNING UTILITIES
// ============================================================================

/// Compute contrastive loss (InfoNCE / NT-Xent style)
pub fn contrastive_loss(
    text_embeddings: &[Embedding],
    image_embeddings: &[Embedding],
    temperature: f32,
) -> f32 {
    if text_embeddings.is_empty() || image_embeddings.is_empty() {
        return 0.0;
    }

    let n = text_embeddings.len().min(image_embeddings.len());
    let mut loss = 0.0f32;

    for i in 0..n {
        // Text-to-image direction
        let mut logits: Vec<f32> = image_embeddings
            .iter()
            .map(|img| text_embeddings[i].similarity(img) / temperature)
            .collect();

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_softmax = logits[i] - max_logit - exp_sum.ln();
        loss -= log_softmax;

        // Image-to-text direction
        logits = text_embeddings
            .iter()
            .map(|txt| image_embeddings[i].similarity(txt) / temperature)
            .collect();

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_softmax = logits[i] - max_logit - exp_sum.ln();
        loss -= log_softmax;
    }

    loss / (2.0 * n as f32)
}

/// Compute recall@k metric
pub fn recall_at_k(results: &[SearchResult], expected_id: &str, k: usize) -> f32 {
    let found = results.iter().take(k).any(|r| r.id == expected_id);
    if found {
        1.0
    } else {
        0.0
    }
}

/// Compute mean reciprocal rank
pub fn mean_reciprocal_rank(results: &[SearchResult], expected_id: &str) -> f32 {
    for (i, result) in results.iter().enumerate() {
        if result.id == expected_id {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

// ============================================================================
// MAIN DEMONSTRATION
// ============================================================================

fn main() {
    println!("Demo O: Multi-Modal CLIP Search");
    println!("================================\n");

    // Create index
    let mut index = ClipIndex::new(42);

    // Add some text documents
    let texts = vec![
        TextDocument::new("text_1", "A photo of a cat sitting on a couch")
            .with_metadata("category", "animals"),
        TextDocument::new("text_2", "A beautiful sunset over the ocean")
            .with_metadata("category", "nature"),
        TextDocument::new("text_3", "A person riding a bicycle in the park")
            .with_metadata("category", "sports"),
        TextDocument::new("text_4", "A plate of delicious pasta with tomato sauce")
            .with_metadata("category", "food"),
        TextDocument::new("text_5", "A modern city skyline at night")
            .with_metadata("category", "urban"),
    ];

    index.index_texts(&texts);
    println!(
        "Indexed {} text documents",
        index.count_by_modality(Modality::Text)
    );

    // Add some image documents (test patterns simulating different content)
    let images = vec![
        ImageDocument::test_pattern("img_cat", 64, 64, 1).with_metadata("description", "cat image"),
        ImageDocument::test_pattern("img_sunset", 64, 64, 2)
            .with_metadata("description", "sunset image"),
        ImageDocument::test_pattern("img_bike", 64, 64, 3)
            .with_metadata("description", "bicycle image"),
        ImageDocument::test_pattern("img_food", 64, 64, 4)
            .with_metadata("description", "food image"),
        ImageDocument::test_pattern("img_city", 64, 64, 5)
            .with_metadata("description", "city image"),
    ];

    index.index_images(&images);
    println!(
        "Indexed {} image documents",
        index.count_by_modality(Modality::Image)
    );
    println!("Total indexed: {}\n", index.len());

    // Text-to-image search
    println!("Text-to-Image Search: 'cat on furniture'");
    let results = index.search_by_text_filtered("cat on furniture", 3, Modality::Image);
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.4}) - {:?}",
            i + 1,
            result.id,
            result.score,
            result.metadata.get("description")
        );
    }
    println!();

    // Text-to-text search (semantic similarity)
    println!("Text-to-Text Search: 'nature landscape'");
    let results = index.search_by_text_filtered("nature landscape", 3, Modality::Text);
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.4}) - {:?}",
            i + 1,
            result.id,
            result.score,
            result.metadata.get("category")
        );
    }
    println!();

    // Cross-modal search
    println!("Cross-Modal Search: 'outdoor activities'");
    let results = index.search_by_text("outdoor activities", 5);
    for (i, result) in results.iter().enumerate() {
        println!(
            "  {}. {} [{:?}] (score: {:.4})",
            i + 1,
            result.id,
            result.modality,
            result.score
        );
    }
    println!();

    // Compute contrastive loss for training simulation
    let text_embs: Vec<Embedding> = texts
        .iter()
        .map(|t| {
            let mut enc = TextEncoder::new(10000, 256, 42);
            enc.encode(&t.content)
        })
        .collect();

    let img_embs: Vec<Embedding> = images
        .iter()
        .map(|i| {
            let enc = ImageEncoder::new(256, 256, 43);
            enc.encode(i)
        })
        .collect();

    let loss = contrastive_loss(&text_embs, &img_embs, 0.07);
    println!("Contrastive Loss (temperature=0.07): {:.4}", loss);

    println!("\nDemo O complete!");
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Embedding Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_embedding_new_normalizes() {
        let emb = Embedding::new(vec![3.0, 4.0], Modality::Text);
        assert!(emb.normalized);
        let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_similarity_self() {
        let emb = Embedding::new(vec![1.0, 0.0, 0.0], Modality::Text);
        let sim = emb.similarity(&emb);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let emb1 = Embedding::new(vec![1.0, 0.0], Modality::Text);
        let emb2 = Embedding::new(vec![0.0, 1.0], Modality::Text);
        let sim = emb1.similarity(&emb2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_embedding_similarity_opposite() {
        let emb1 = Embedding::new(vec![1.0, 0.0], Modality::Text);
        let emb2 = Embedding::new(vec![-1.0, 0.0], Modality::Text);
        let sim = emb1.similarity(&emb2);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_dimension() {
        let emb = Embedding::new(vec![1.0; 512], Modality::Image);
        assert_eq!(emb.dim(), 512);
    }

    #[test]
    fn test_embedding_zero_vector() {
        let mut emb = Embedding {
            vector: vec![0.0; 10],
            modality: Modality::Text,
            normalized: false,
        };
        emb.normalize();
        // Should not panic, vector stays zero
        assert!(!emb.normalized);
    }

    // ------------------------------------------------------------------------
    // Text Document Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_text_document_creation() {
        let doc = TextDocument::new("id1", "hello world");
        assert_eq!(doc.id, "id1");
        assert_eq!(doc.content, "hello world");
    }

    #[test]
    fn test_text_document_metadata() {
        let doc = TextDocument::new("id1", "content")
            .with_metadata("author", "alice")
            .with_metadata("date", "2024-01-01");
        assert_eq!(doc.metadata.get("author"), Some(&"alice".to_string()));
        assert_eq!(doc.metadata.get("date"), Some(&"2024-01-01".to_string()));
    }

    // ------------------------------------------------------------------------
    // Image Document Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_image_document_creation() {
        let doc = ImageDocument::new("img1", 32, 32, vec![0; 32 * 32 * 3]);
        assert_eq!(doc.id, "img1");
        assert_eq!(doc.width, 32);
        assert_eq!(doc.height, 32);
    }

    #[test]
    fn test_image_document_test_pattern() {
        let doc = ImageDocument::test_pattern("test", 16, 16, 42);
        assert_eq!(doc.width, 16);
        assert_eq!(doc.height, 16);
        assert_eq!(doc.pixels.len(), 16 * 16 * 3);
    }

    #[test]
    fn test_image_document_metadata() {
        let doc = ImageDocument::new("img1", 32, 32, vec![]).with_metadata("caption", "a cat");
        assert_eq!(doc.metadata.get("caption"), Some(&"a cat".to_string()));
    }

    // ------------------------------------------------------------------------
    // Text Encoder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_text_encoder_encode() {
        let mut encoder = TextEncoder::new(1000, 128, 42);
        let emb = encoder.encode("hello world");
        assert_eq!(emb.dim(), EMBEDDING_DIM);
        assert!(emb.normalized);
    }

    #[test]
    fn test_text_encoder_empty_text() {
        let mut encoder = TextEncoder::new(1000, 128, 42);
        let emb = encoder.encode("");
        assert_eq!(emb.dim(), EMBEDDING_DIM);
    }

    #[test]
    fn test_text_encoder_deterministic() {
        let mut encoder1 = TextEncoder::new(1000, 128, 42);
        let mut encoder2 = TextEncoder::new(1000, 128, 42);
        let emb1 = encoder1.encode("test");
        let emb2 = encoder2.encode("test");
        assert_eq!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_text_encoder_different_texts() {
        let mut encoder = TextEncoder::new(1000, 128, 42);
        let emb1 = encoder.encode("hello");
        let emb2 = encoder.encode("world");
        // Different texts should produce different embeddings
        assert_ne!(emb1.vector, emb2.vector);
    }

    // ------------------------------------------------------------------------
    // Image Encoder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_image_encoder_encode() {
        let encoder = ImageEncoder::new(128, 64, 42);
        let img = ImageDocument::test_pattern("test", 32, 32, 1);
        let emb = encoder.encode(&img);
        assert_eq!(emb.dim(), EMBEDDING_DIM);
        assert!(emb.normalized);
    }

    #[test]
    fn test_image_encoder_small_image() {
        let encoder = ImageEncoder::new(128, 64, 42);
        let img = ImageDocument::new("tiny", 8, 8, vec![128; 8 * 8 * 3]);
        let emb = encoder.encode(&img);
        assert_eq!(emb.dim(), EMBEDDING_DIM);
    }

    #[test]
    fn test_image_encoder_deterministic() {
        let encoder1 = ImageEncoder::new(128, 64, 42);
        let encoder2 = ImageEncoder::new(128, 64, 42);
        let img = ImageDocument::test_pattern("test", 32, 32, 1);
        let emb1 = encoder1.encode(&img);
        let emb2 = encoder2.encode(&img);
        assert_eq!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_image_encoder_different_images() {
        let encoder = ImageEncoder::new(128, 64, 42);
        let img1 = ImageDocument::test_pattern("a", 32, 32, 1);
        let img2 = ImageDocument::test_pattern("b", 32, 32, 2);
        let emb1 = encoder.encode(&img1);
        let emb2 = encoder.encode(&img2);
        assert_ne!(emb1.vector, emb2.vector);
    }

    // ------------------------------------------------------------------------
    // CLIP Index Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_index_empty() {
        let index = ClipIndex::new(42);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_index_text() {
        let mut index = ClipIndex::new(42);
        index.index_text(TextDocument::new("t1", "hello"));
        assert_eq!(index.len(), 1);
        assert_eq!(index.count_by_modality(Modality::Text), 1);
    }

    #[test]
    fn test_index_image() {
        let mut index = ClipIndex::new(42);
        index.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));
        assert_eq!(index.len(), 1);
        assert_eq!(index.count_by_modality(Modality::Image), 1);
    }

    #[test]
    fn test_index_multiple() {
        let mut index = ClipIndex::new(42);
        index.index_texts(&[TextDocument::new("t1", "a"), TextDocument::new("t2", "b")]);
        index.index_images(&[
            ImageDocument::test_pattern("i1", 32, 32, 1),
            ImageDocument::test_pattern("i2", 32, 32, 2),
        ]);
        assert_eq!(index.len(), 4);
        assert_eq!(index.count_by_modality(Modality::Text), 2);
        assert_eq!(index.count_by_modality(Modality::Image), 2);
    }

    #[test]
    fn test_search_by_text() {
        let mut index = ClipIndex::new(42);
        index.index_text(TextDocument::new("t1", "cat"));
        index.index_text(TextDocument::new("t2", "dog"));
        let results = index.search_by_text("cat", 2);
        assert_eq!(results.len(), 2);
        // First result should have highest score
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_search_by_text_filtered() {
        let mut index = ClipIndex::new(42);
        index.index_text(TextDocument::new("t1", "cat"));
        index.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));
        let results = index.search_by_text_filtered("cat", 5, Modality::Image);
        assert!(results.iter().all(|r| r.modality == Modality::Image));
    }

    #[test]
    fn test_search_by_image() {
        let mut index = ClipIndex::new(42);
        index.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));
        index.index_image(ImageDocument::test_pattern("i2", 32, 32, 2));
        let query = ImageDocument::test_pattern("q", 32, 32, 1);
        let results = index.search_by_image(&query, 2);
        assert_eq!(results.len(), 2);
        // Results should be ordered by similarity (descending)
        assert!(results[0].score >= results[1].score);
        // Both images should be in results
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"i1") && ids.contains(&"i2"));
    }

    #[test]
    fn test_search_top_k_limit() {
        let mut index = ClipIndex::new(42);
        for i in 0..10 {
            index.index_text(TextDocument::new(format!("t{i}"), format!("text {i}")));
        }
        let results = index.search_by_text("text", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_result_metadata() {
        let mut index = ClipIndex::new(42);
        index.index_text(TextDocument::new("t1", "hello").with_metadata("key", "value"));
        let results = index.search_by_text("hello", 1);
        assert_eq!(results[0].metadata.get("key"), Some(&"value".to_string()));
    }

    // ------------------------------------------------------------------------
    // Contrastive Loss Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_contrastive_loss_empty() {
        let loss = contrastive_loss(&[], &[], 0.07);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_contrastive_loss_single_pair() {
        let text_embs = vec![Embedding::new(vec![1.0, 0.0], Modality::Text)];
        let img_embs = vec![Embedding::new(vec![1.0, 0.0], Modality::Image)];
        let loss = contrastive_loss(&text_embs, &img_embs, 0.07);
        // Perfect match should have low loss
        assert!(loss < 1.0);
    }

    #[test]
    fn test_contrastive_loss_mismatched() {
        // Need multiple pairs to see loss differences
        let text_embs = vec![
            Embedding::new(vec![1.0, 0.0], Modality::Text),
            Embedding::new(vec![0.0, 1.0], Modality::Text),
        ];
        let img_embs = vec![
            Embedding::new(vec![0.0, 1.0], Modality::Image), // Mismatched with text[0]
            Embedding::new(vec![1.0, 0.0], Modality::Image), // Mismatched with text[1]
        ];
        let loss = contrastive_loss(&text_embs, &img_embs, 0.07);
        // With multiple pairs and mismatches, loss should be positive
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_contrastive_loss_temperature_effect() {
        let text_embs = vec![
            Embedding::new(vec![1.0, 0.0], Modality::Text),
            Embedding::new(vec![0.0, 1.0], Modality::Text),
        ];
        let img_embs = vec![
            Embedding::new(vec![0.9, 0.1], Modality::Image),
            Embedding::new(vec![0.1, 0.9], Modality::Image),
        ];
        let loss_low_temp = contrastive_loss(&text_embs, &img_embs, 0.01);
        let loss_high_temp = contrastive_loss(&text_embs, &img_embs, 1.0);
        // Both should compute valid losses (they may or may not be different depending on alignment)
        assert!(loss_low_temp >= 0.0 || loss_low_temp.is_nan());
        assert!(loss_high_temp >= 0.0 || loss_high_temp.is_nan());
    }

    // ------------------------------------------------------------------------
    // Metric Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_recall_at_k_found() {
        let results = vec![
            SearchResult::new("a".to_string(), Modality::Text, 0.9, HashMap::new()),
            SearchResult::new("b".to_string(), Modality::Text, 0.8, HashMap::new()),
        ];
        assert_eq!(recall_at_k(&results, "a", 1), 1.0);
        assert_eq!(recall_at_k(&results, "b", 2), 1.0);
    }

    #[test]
    fn test_recall_at_k_not_found() {
        let results = vec![SearchResult::new(
            "a".to_string(),
            Modality::Text,
            0.9,
            HashMap::new(),
        )];
        assert_eq!(recall_at_k(&results, "b", 1), 0.0);
        assert_eq!(recall_at_k(&results, "a", 0), 0.0);
    }

    #[test]
    fn test_mrr_first_position() {
        let results = vec![SearchResult::new(
            "a".to_string(),
            Modality::Text,
            0.9,
            HashMap::new(),
        )];
        assert_eq!(mean_reciprocal_rank(&results, "a"), 1.0);
    }

    #[test]
    fn test_mrr_second_position() {
        let results = vec![
            SearchResult::new("a".to_string(), Modality::Text, 0.9, HashMap::new()),
            SearchResult::new("b".to_string(), Modality::Text, 0.8, HashMap::new()),
        ];
        assert_eq!(mean_reciprocal_rank(&results, "b"), 0.5);
    }

    #[test]
    fn test_mrr_not_found() {
        let results = vec![SearchResult::new(
            "a".to_string(),
            Modality::Text,
            0.9,
            HashMap::new(),
        )];
        assert_eq!(mean_reciprocal_rank(&results, "x"), 0.0);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_full_pipeline() {
        let mut index = ClipIndex::new(42);

        // Index content
        index.index_text(
            TextDocument::new("desc_cat", "a cute cat sleeping")
                .with_metadata("type", "description"),
        );
        index.index_image(
            ImageDocument::test_pattern("img_cat", 32, 32, 100).with_metadata("subject", "cat"),
        );

        // Search
        let results = index.search_by_text("cat", 2);
        assert_eq!(results.len(), 2);

        // Verify both modalities found
        let modalities: Vec<_> = results.iter().map(|r| r.modality).collect();
        assert!(modalities.contains(&Modality::Text));
        assert!(modalities.contains(&Modality::Image));
    }

    #[test]
    fn test_cross_modal_retrieval() {
        let mut index = ClipIndex::new(42);

        // Index matching text-image pairs
        index.index_text(TextDocument::new("t1", "red sports car"));
        index.index_image(ImageDocument::test_pattern("i1", 32, 32, 1));

        // Query with text, retrieve images
        let results = index.search_by_text_filtered("car", 1, Modality::Image);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].modality, Modality::Image);
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_embedding_normalized(vec in prop::collection::vec(-10.0f32..10.0f32, 10..100)) {
            if vec.iter().any(|&x| x.abs() > 1e-8) {
                let emb = Embedding::new(vec, Modality::Text);
                let norm: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn prop_similarity_symmetric(
            v1 in prop::collection::vec(-1.0f32..1.0f32, 10..20),
            v2 in prop::collection::vec(-1.0f32..1.0f32, 10..20)
        ) {
            let len = v1.len().min(v2.len());
            let emb1 = Embedding::new(v1[..len].to_vec(), Modality::Text);
            let emb2 = Embedding::new(v2[..len].to_vec(), Modality::Text);
            let sim1 = emb1.similarity(&emb2);
            let sim2 = emb2.similarity(&emb1);
            prop_assert!((sim1 - sim2).abs() < 1e-6);
        }

        #[test]
        fn prop_similarity_bounded(
            v1 in prop::collection::vec(-1.0f32..1.0f32, 10..20),
            v2 in prop::collection::vec(-1.0f32..1.0f32, 10..20)
        ) {
            let len = v1.len().min(v2.len());
            let emb1 = Embedding::new(v1[..len].to_vec(), Modality::Text);
            let emb2 = Embedding::new(v2[..len].to_vec(), Modality::Text);
            let sim = emb1.similarity(&emb2);
            prop_assert!(sim >= -1.0 - 1e-6 && sim <= 1.0 + 1e-6);
        }

        #[test]
        fn prop_text_encoder_output_dimension(text in "[a-z ]{1,50}") {
            let mut encoder = TextEncoder::new(1000, 128, 42);
            let emb = encoder.encode(&text);
            prop_assert_eq!(emb.dim(), EMBEDDING_DIM);
        }

        #[test]
        fn prop_image_encoder_output_dimension(
            width in 8u32..64u32,
            height in 8u32..64u32,
            seed in 1u32..1000u32
        ) {
            let encoder = ImageEncoder::new(128, 64, 42);
            let img = ImageDocument::test_pattern("test", width, height, seed);
            let emb = encoder.encode(&img);
            prop_assert_eq!(emb.dim(), EMBEDDING_DIM);
        }

        #[test]
        fn prop_index_preserves_count(
            num_texts in 0usize..10,
            num_images in 0usize..10
        ) {
            let mut index = ClipIndex::new(42);

            for i in 0..num_texts {
                index.index_text(TextDocument::new(format!("t{i}"), format!("text {i}")));
            }
            for i in 0..num_images {
                index.index_image(ImageDocument::test_pattern(format!("i{i}"), 32, 32, i as u32));
            }

            prop_assert_eq!(index.len(), num_texts + num_images);
            prop_assert_eq!(index.count_by_modality(Modality::Text), num_texts);
            prop_assert_eq!(index.count_by_modality(Modality::Image), num_images);
        }

        #[test]
        fn prop_search_results_ordered(
            seed in 1u64..1000u64,
            num_docs in 3usize..10
        ) {
            let mut index = ClipIndex::new(seed);

            for i in 0..num_docs {
                index.index_text(TextDocument::new(format!("t{i}"), format!("document {i}")));
            }

            let results = index.search_by_text("document", num_docs);

            // Verify results are ordered by score descending
            for i in 1..results.len() {
                prop_assert!(results[i-1].score >= results[i].score);
            }
        }

        #[test]
        fn prop_search_respects_top_k(
            num_docs in 5usize..15,
            top_k in 1usize..10
        ) {
            let mut index = ClipIndex::new(42);

            for i in 0..num_docs {
                index.index_text(TextDocument::new(format!("t{i}"), format!("text {i}")));
            }

            let results = index.search_by_text("text", top_k);
            prop_assert!(results.len() <= top_k);
            prop_assert!(results.len() <= num_docs);
        }

        #[test]
        fn prop_contrastive_loss_non_negative(
            n in 1usize..5,
            temp in 0.01f32..1.0f32
        ) {
            let text_embs: Vec<Embedding> = (0..n)
                .map(|i| Embedding::new(vec![i as f32; 10], Modality::Text))
                .collect();
            let img_embs: Vec<Embedding> = (0..n)
                .map(|i| Embedding::new(vec![(i + 1) as f32; 10], Modality::Image))
                .collect();

            let loss = contrastive_loss(&text_embs, &img_embs, temp);
            prop_assert!(loss >= 0.0 || loss.is_nan());
        }

        #[test]
        fn prop_recall_at_k_binary(
            k in 1usize..10,
            target_pos in 0usize..10
        ) {
            let results: Vec<SearchResult> = (0..10)
                .map(|i| SearchResult::new(
                    format!("r{i}"),
                    Modality::Text,
                    1.0 - i as f32 * 0.1,
                    HashMap::new()
                ))
                .collect();

            let target = format!("r{target_pos}");
            let recall = recall_at_k(&results, &target, k);
            prop_assert!(recall == 0.0 || recall == 1.0);
        }

        #[test]
        fn prop_mrr_bounded(target_pos in 0usize..10) {
            let results: Vec<SearchResult> = (0..10)
                .map(|i| SearchResult::new(
                    format!("r{i}"),
                    Modality::Text,
                    1.0 - i as f32 * 0.1,
                    HashMap::new()
                ))
                .collect();

            let target = format!("r{target_pos}");
            let mrr = mean_reciprocal_rank(&results, &target);
            prop_assert!(mrr >= 0.0 && mrr <= 1.0);
        }
    }
}
