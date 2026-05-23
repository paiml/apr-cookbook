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
use std::time::Instant;

/// Block size for FlashAttention tiling
pub const BLOCK_SIZE: usize = 64;

/// Compute backend with automatic fallback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    // GPU via wgpu/CUDA
    Gpu,
    // SIMD via trueno (AVX-512/AVX2/NEON)
    Simd,
    // Pure Rust scalar
    Scalar,
}

impl ComputeBackend {
    /// Detect best available backend
    pub fn detect() -> Self {
        // Check GPU availability (simulated)
        if Self::gpu_available() {
            return Self::Gpu;
        }

        // Check SIMD availability
        if Self::simd_available() {
            return Self::Simd;
        }

        Self::Scalar
    }

    pub fn gpu_available() -> bool {
        // In real implementation, check wgpu adapter availability
        // For demo, check environment variable
        std::env::var("APR_GPU_ENABLED").is_ok()
    }

    pub fn simd_available() -> bool {
        // Check for AVX2 or NEON
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            true // NEON is always available on aarch64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Get theoretical peak GFLOPS for this backend
    pub fn peak_gflops(self) -> f64 {
        match self {
            Self::Gpu => 10000.0, // Modern GPU (simplified)
            Self::Simd => 100.0,  // AVX-512 on modern CPU
            Self::Scalar => 10.0, // Single-threaded scalar
        }
    }
}

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    // Sequence length
    pub seq_len: usize,
    // Hidden dimension
    pub d_model: usize,
    // Number of attention heads
    pub n_heads: usize,
    // Head dimension (d_model / n_heads)
    pub d_head: usize,
    // Use FlashAttention algorithm
    pub use_flash: bool,
}

impl AttentionConfig {
    pub fn new(seq_len: usize, d_model: usize, n_heads: usize) -> Self {
        Self {
            seq_len,
            d_model,
            n_heads,
            d_head: d_model / n_heads,
            use_flash: true,
        }
    }

    /// Calculate FLOPs for attention computation
    pub fn flops(&self) -> usize {
        // QK^T: 2 * seq_len * seq_len * d_head (per head)
        // softmax: ~5 * seq_len * seq_len (per head)
        // AV: 2 * seq_len * seq_len * d_head (per head)
        let per_head =
            4 * self.seq_len * self.seq_len * self.d_head + 5 * self.seq_len * self.seq_len;
        per_head * self.n_heads
    }

    /// Calculate memory for standard attention (O(N²))
    pub fn standard_memory_bytes(&self) -> usize {
        // QKV: 3 * seq_len * d_model * 4 bytes
        // Attention scores: seq_len * seq_len * n_heads * 4 bytes
        3 * self.seq_len * self.d_model * 4 + self.seq_len * self.seq_len * self.n_heads * 4
    }

    /// Calculate memory for FlashAttention (O(N))
    pub fn flash_memory_bytes(&self) -> usize {
        // QKV: 3 * seq_len * d_model * 4 bytes
        // Tile buffer: BLOCK_SIZE * BLOCK_SIZE * n_heads * 4 bytes
        3 * self.seq_len * self.d_model * 4 + BLOCK_SIZE * BLOCK_SIZE * self.n_heads * 4
    }
}

/// Attention computation result
#[derive(Debug)]
pub struct AttentionResult {
    // Output tensor (flattened)
    pub output: Vec<f32>,
    // Computation time in milliseconds
    pub time_ms: f64,
    // Achieved GFLOPS
    pub gflops: f64,
    // Backend used
    pub backend: ComputeBackend,
    // Memory used in bytes
    pub memory_bytes: usize,
}

// Compute dot product between row `a_row` of `a` and row `b_row` of `b`,
/// each with stride `d_head`.
pub fn dot_product(a: &[f32], a_row: usize, b: &[f32], b_row: usize, d_head: usize) -> f32 {
    let a_off = a_row * d_head;
    let b_off = b_row * d_head;
    (0..d_head).map(|d| a[a_off + d] * b[b_off + d]).sum()
}

/// Compute softmax-normalised attention scores for a single query row.
pub fn softmax_scores(
    q: &[f32],
    k: &[f32],
    row: usize,
    seq_len: usize,
    d_head: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores: Vec<f32> = (0..seq_len)
        .map(|j| dot_product(q, row, k, j, d_head) * scale)
        .collect();

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    for s in &mut scores {
        *s /= sum;
    }
    scores
}

/// Accumulate weighted values into `output` for a single query row.
pub fn accumulate_weighted_values(
    output: &mut [f32],
    scores: &[f32],
    v: &[f32],
    row: usize,
    seq_len: usize,
    d_head: usize,
) {
    let out_off = row * d_head;
    for (j, &w) in scores.iter().enumerate().take(seq_len) {
        let v_off = j * d_head;
        for d in 0..d_head {
            output[out_off + d] += w * v[v_off + d];
        }
    }
}

/// Standard (naive) attention computation
pub fn naive_attention(q: &[f32], k: &[f32], v: &[f32], config: &AttentionConfig) -> Vec<f32> {
    let seq_len = config.seq_len;
    let d_head = config.d_head;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut output = vec![0.0f32; seq_len * d_head];

    for i in 0..seq_len {
        let scores = softmax_scores(q, k, i, seq_len, d_head, scale);
        accumulate_weighted_values(&mut output, &scores, v, i, seq_len, d_head);
    }

    output
}

// Process one row of query `i` against key/value tile `j_start..j_end`,
/// updating `output`, `row_max`, and `row_sum` with online softmax.
#[allow(clippy::too_many_arguments)]
pub fn flash_tile_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    row_max: &mut f32,
    row_sum: &mut f32,
    i: usize,
    j_start: usize,
    j_end: usize,
    d_head: usize,
    scale: f32,
) {
    let old_max = *row_max;
    let old_sum = *row_sum;

    // Compute scores for this tile
    let mut tile_scores: Vec<f32> = (j_start..j_end)
        .map(|j| dot_product(q, i, k, j, d_head) * scale)
        .collect();
    let tile_max = tile_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Online softmax update
    let new_max = old_max.max(tile_max);
    let old_rescale = (old_max - new_max).exp();
    let tile_rescale = (tile_max - new_max).exp();

    let mut tile_sum = 0.0f32;
    for score in &mut tile_scores {
        *score = (*score - tile_max).exp() * tile_rescale;
        tile_sum += *score;
    }

    let new_sum = old_sum * old_rescale + tile_sum;

    // Rescale existing output
    let out_off = i * d_head;
    let rescale = old_sum * old_rescale / new_sum;
    for d in 0..d_head {
        output[out_off + d] *= rescale;
    }

    // Add contribution from this tile
    for (tj, j) in (j_start..j_end).enumerate() {
        let weight = tile_scores[tj] / new_sum;
        let v_off = j * d_head;
        for d in 0..d_head {
            output[out_off + d] += weight * v[v_off + d];
        }
    }

    *row_max = new_max;
    *row_sum = new_sum;
}

/// FlashAttention-style tiled attention (memory efficient)
pub fn flash_attention(q: &[f32], k: &[f32], v: &[f32], config: &AttentionConfig) -> Vec<f32> {
    let seq_len = config.seq_len;
    let d_head = config.d_head;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut output = vec![0.0f32; seq_len * d_head];
    let mut row_max = vec![f32::NEG_INFINITY; seq_len];
    let mut row_sum = vec![0.0f32; seq_len];

    // Process in tiles for memory efficiency
    let n_blocks = seq_len.div_ceil(BLOCK_SIZE);

    for block_j in 0..n_blocks {
        let j_start = block_j * BLOCK_SIZE;
        let j_end = (j_start + BLOCK_SIZE).min(seq_len);

        for block_i in 0..n_blocks {
            let i_start = block_i * BLOCK_SIZE;
            let i_end = (i_start + BLOCK_SIZE).min(seq_len);

            for i in i_start..i_end {
                flash_tile_row(
                    q,
                    k,
                    v,
                    &mut output,
                    &mut row_max[i],
                    &mut row_sum[i],
                    i,
                    j_start,
                    j_end,
                    d_head,
                    scale,
                );
            }
        }
    }

    output
}

/// Run attention with specified backend
pub fn run_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &AttentionConfig,
    backend: ComputeBackend,
) -> AttentionResult {
    let start = Instant::now();

    let output = if config.use_flash {
        flash_attention(q, k, v, config)
    } else {
        naive_attention(q, k, v, config)
    };

    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0;
    let flops = config.flops() as f64;
    let gflops = flops / (elapsed.as_secs_f64() * 1e9);

    let memory_bytes = if config.use_flash {
        config.flash_memory_bytes()
    } else {
        config.standard_memory_bytes()
    };

    AttentionResult {
        output,
        time_ms,
        gflops,
        backend,
        memory_bytes,
    }
}

/// Generate random tensor for testing
pub fn generate_tensor(seed: u64, size: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = vec![0.0f32; size];
    for (i, val) in data.iter_mut().enumerate() {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash = hasher.finish();
        *val = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
    }
    data
}
