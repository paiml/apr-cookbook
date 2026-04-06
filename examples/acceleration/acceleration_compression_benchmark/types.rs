#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::time::Instant;

/// Compression method to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    None,
    Lz4,
    Zstd(i32),
}

impl CompressionMethod {
    pub fn name(self) -> String {
        match self {
            Self::None => "None".to_string(),
            Self::Lz4 => "LZ4".to_string(),
            Self::Zstd(level) => format!("ZSTD-{level}"),
        }
    }
}

/// Results from benchmarking a single compression method on a payload.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub method_name: String,
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
    pub compress_time_ms: f64,
    pub decompress_time_ms: f64,
    pub compress_throughput_gbps: f64,
    pub decompress_throughput_gbps: f64,
}

/// Compress `data` using the specified method.
pub fn compress_data(data: &[u8], method: CompressionMethod) -> Result<Vec<u8>, String> {
    match method {
        CompressionMethod::None => Ok(data.to_vec()),
        CompressionMethod::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
        CompressionMethod::Zstd(level) => {
            zstd::encode_all(std::io::Cursor::new(data), level).map_err(|e| e.to_string())
        }
    }
}

/// Decompress `compressed` using the specified method.
pub fn decompress_data(compressed: &[u8], method: CompressionMethod) -> Result<Vec<u8>, String> {
    match method {
        CompressionMethod::None => Ok(compressed.to_vec()),
        CompressionMethod::Lz4 => {
            lz4_flex::decompress_size_prepended(compressed).map_err(|e| e.to_string())
        }
        CompressionMethod::Zstd(_) => {
            zstd::decode_all(std::io::Cursor::new(compressed)).map_err(|e| e.to_string())
        }
    }
}

/// Run `iterations` rounds of compress + decompress, return aggregated metrics.
pub fn benchmark_compression(
    data: &[u8],
    method: CompressionMethod,
    iterations: usize,
) -> CompressionResult {
    let original_size = data.len();

    // Warmup
    let _ = compress_data(data, method);

    // Benchmark compression
    let start = Instant::now();
    let mut compressed = Vec::new();
    for _ in 0..iterations {
        compressed = compress_data(data, method).unwrap_or_default();
    }
    let compress_elapsed = start.elapsed();
    let compressed_size = compressed.len();

    // Benchmark decompression
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = decompress_data(&compressed, method);
    }
    let decompress_elapsed = start.elapsed();

    let compress_ms = compress_elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let decompress_ms = decompress_elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let gb = original_size as f64 / 1_073_741_824.0;

    CompressionResult {
        method_name: method.name(),
        original_size,
        compressed_size,
        ratio: if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        },
        compress_time_ms: compress_ms,
        decompress_time_ms: decompress_ms,
        compress_throughput_gbps: if compress_ms > 0.0 {
            gb / (compress_ms / 1000.0)
        } else {
            0.0
        },
        decompress_throughput_gbps: if decompress_ms > 0.0 {
            gb / (decompress_ms / 1000.0)
        } else {
            0.0
        },
    }
}

// -- Data Generators ----------------------------------------------------------

/// Generate pseudo-random bytes using a deterministic hash-based PRNG.
pub fn generate_random_data(size: usize, seed: u64) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        data.push((hasher.finish() & 0xFF) as u8);
    }
    data
}

// Simulate .apr tensor weights: 4 KB blocks with a repeating 64-byte pattern
// and low-magnitude perturbations (2-bit noise). Real model weights have strong
/// local correlation that compression algorithms exploit.
pub fn generate_model_like_data(size: usize, seed: u64) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut data = Vec::with_capacity(size);
    let block_size = 4096;
    let pattern_len = 64;
    let mut i = 0;
    while i < size {
        let mut hasher = DefaultHasher::new();
        (seed, i / block_size).hash(&mut hasher);
        let block_seed = hasher.finish();
        let mut pattern = [0u8; 64];
        for (p, slot) in pattern.iter_mut().enumerate() {
            let mut h = DefaultHasher::new();
            (block_seed, p).hash(&mut h);
            *slot = (h.finish() & 0xFF) as u8;
        }
        let end = (i + block_size).min(size);
        for j in i..end {
            let base = pattern[(j - i) % pattern_len];
            let mut h2 = DefaultHasher::new();
            (seed, j).hash(&mut h2);
            data.push(base ^ (h2.finish() & 0x03) as u8);
        }
        i = end;
    }
    data
}

// Sparse data simulating pruned model weights. `sparsity` fraction of bytes
/// are zero; the rest are pseudo-random.
pub fn generate_sparse_data(size: usize, seed: u64, sparsity: f64) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut data = Vec::with_capacity(size);
    let threshold = (sparsity * 256.0) as u64;
    for i in 0..size {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash = hasher.finish();
        if ((hash >> 32) & 0xFF) < threshold {
            data.push(0u8);
        } else {
            data.push((hash & 0xFF) as u8);
        }
    }
    data
}

// -- Display Helpers ----------------------------------------------------------

pub fn print_result_header() {
    println!(
        "   {:<8} {:>10} {:>10} {:>7} {:>9} {:>9} {:>10} {:>10}",
        "Method", "Original", "Compress", "Ratio", "Comp ms", "Dec ms", "Comp GB/s", "Dec GB/s"
    );
    println!("   {}", "-".repeat(78));
}

pub fn print_result_row(r: &CompressionResult) {
    println!(
        "   {:<8} {:>10} {:>10} {:>6.2}x {:>8.2} {:>9.2} {:>9.2} {:>10.2}",
        r.method_name,
        fmt_bytes(r.original_size),
        fmt_bytes(r.compressed_size),
        r.ratio,
        r.compress_time_ms,
        r.decompress_time_ms,
        r.compress_throughput_gbps,
        r.decompress_throughput_gbps,
    );
}

pub fn fmt_bytes(n: usize) -> String {
    if n >= 1_048_576 {
        format!("{:.1} MB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

// -- Main ---------------------------------------------------------------------
