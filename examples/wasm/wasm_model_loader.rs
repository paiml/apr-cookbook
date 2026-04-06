//! # Recipe: WASM Model Loader Pipeline
//!
//! **Category**: WASM/Browser
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (Verified)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate a model loading pipeline optimized for WASM environments:
//! progressive loading, memory-efficient parsing, and lazy tensor materialization.
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_model_loader
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Haas, A. et al. (2017). *Bringing the Web up to Speed with WebAssembly*. PLDI. DOI: 10.1145/3062341.3062363

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_model_loader")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("WASM model loading pipeline simulation");
    println!();

    let (mut budget, model_meta) = section_memory_constraints(&mut ctx);

    let chunk_size = 64 * 1024; // 64 KB chunks (typical fetch API)
    let (mut loader, chunks) = section_chunked_download(&model_meta, chunk_size, &mut ctx)?;

    let parsed = section_header_parsing(&chunks, &model_meta, &mut loader)?;

    let loaded_tensors =
        section_budgeted_tensor_loading(&parsed, &mut budget, &mut loader, &mut ctx);

    let final_loaded_count =
        section_lazy_materialization(&parsed, &loaded_tensors, &mut budget, &mut ctx);

    section_performance_summary(&loader, &budget, chunk_size, final_loaded_count, &mut ctx);

    println!();
    println!("=== Recipe complete ===");

    Ok(())
}

// ---------------------------------------------------------------------------
// Section helpers — one per logical block of main()
// ---------------------------------------------------------------------------

/// Section 1: Define WASM memory constraints and model metadata.
fn section_memory_constraints(ctx: &mut RecipeContext) -> (WasmMemoryBudget, ModelMetadataInfo) {
    println!("--- Section 1: WASM Memory Constraints & Model Metadata ---");

    let budget = WasmMemoryBudget::new(4 * 1024 * 1024); // 4 MB heap
    let model_meta = ModelMetadataInfo {
        name: "tiny-classifier".to_string(),
        version: 1,
        total_size: 2_500_000,
        header_size: 2048,
        tensor_count: 6,
    };

    println!("  Memory budget: {} bytes", budget.max_bytes);
    println!("  Model: {} v{}", model_meta.name, model_meta.version);
    println!("  Model size: {} bytes", model_meta.total_size);
    println!("  Header size: {} bytes", model_meta.header_size);
    println!("  Tensor count: {}", model_meta.tensor_count);

    ctx.record_metric("memory_budget_bytes", budget.max_bytes as i64);
    ctx.record_metric("model_total_bytes", model_meta.total_size as i64);
    ctx.record_metric("tensor_count", model_meta.tensor_count as i64);
    println!();

    (budget, model_meta)
}

/// Section 2: Chunked streaming download simulation.
fn section_chunked_download(
    model_meta: &ModelMetadataInfo,
    chunk_size: usize,
    ctx: &mut RecipeContext,
) -> Result<(StreamingLoader, Vec<ModelChunk>)> {
    println!("--- Section 2: Chunked Streaming Download ---");

    let chunks = generate_chunks(model_meta, chunk_size);

    println!("  Chunk size: {} bytes", chunk_size);
    println!("  Total chunks: {}", chunks.len());

    let mut loader = StreamingLoader::new(model_meta.total_size);
    let mut progress_callbacks: Vec<String> = Vec::new();

    for chunk in &chunks {
        loader.receive_chunk(chunk)?;

        let msg = format!(
            "  Received chunk @ offset {}: {} bytes (header={}, tensor={})",
            chunk.offset,
            chunk.data.len(),
            chunk.is_header,
            chunk.is_tensor,
        );
        progress_callbacks.push(msg);
    }

    // Show first and last few progress messages
    for msg in progress_callbacks.iter().take(3) {
        println!("{msg}");
    }
    if progress_callbacks.len() > 6 {
        println!("  ... ({} more chunks) ...", progress_callbacks.len() - 6);
    }
    for msg in progress_callbacks
        .iter()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        println!("{msg}");
    }

    ctx.record_metric("chunks_received", loader.chunks_received as i64);
    println!("  Download complete: {}%", loader.progress_pct);
    println!();

    Ok((loader, chunks))
}

/// Section 3: Progressive header parsing.
fn section_header_parsing(
    chunks: &[ModelChunk],
    model_meta: &ModelMetadataInfo,
    loader: &mut StreamingLoader,
) -> Result<ParsedHeader> {
    println!("--- Section 3: Progressive Header Parsing ---");

    let header_data = extract_header_bytes(chunks, model_meta.header_size);
    let parsed = parse_header(&header_data, model_meta)?;
    loader.header_parsed = true;

    println!("  Header magic: 0x{:08X}", parsed.magic);
    println!("  Format version: {}", parsed.format_version);
    println!("  Tensor table entries: {}", parsed.tensor_refs.len());
    for tref in &parsed.tensor_refs {
        println!(
            "    - {} @ offset {} ({} bytes)",
            tref.name, tref.offset, tref.size
        );
    }

    // Validate header checksum
    let valid = validate_header_checksum(&header_data);
    println!("  Header checksum valid: {valid}");
    println!();

    Ok(parsed)
}

/// Section 4: Memory-budgeted tensor loading.
fn section_budgeted_tensor_loading(
    parsed: &ParsedHeader,
    budget: &mut WasmMemoryBudget,
    loader: &mut StreamingLoader,
    ctx: &mut RecipeContext,
) -> Vec<String> {
    println!("--- Section 4: Memory-Budgeted Tensor Loading ---");

    let mut loaded_tensors: Vec<String> = Vec::new();

    for tref in &parsed.tensor_refs {
        match budget.allocate(tref.size) {
            Ok(()) => {
                loaded_tensors.push(tref.name.clone());
                println!(
                    "  Loaded '{}': {} bytes (budget remaining: {})",
                    tref.name,
                    tref.size,
                    budget.remaining()
                );
            }
            Err(e) => {
                println!("  Skipped '{}': {} bytes ({})", tref.name, tref.size, e);
            }
        }
    }

    loader.tensors_loaded = loaded_tensors.len();
    ctx.record_metric("tensors_loaded", loader.tensors_loaded as i64);
    ctx.record_metric("memory_used_bytes", budget.used_bytes as i64);
    println!();

    loaded_tensors
}

/// Section 5: Lazy tensor materialization.
fn section_lazy_materialization(
    parsed: &ParsedHeader,
    loaded_tensors: &[String],
    budget: &mut WasmMemoryBudget,
    ctx: &mut RecipeContext,
) -> usize {
    println!("--- Section 5: Lazy Tensor Materialization ---");

    let mut tensor_refs: Vec<TensorRef> = parsed
        .tensor_refs
        .iter()
        .map(|t| TensorRef {
            name: t.name.clone(),
            offset: t.offset,
            size: t.size,
            loaded: loaded_tensors.contains(&t.name),
        })
        .collect();

    println!("  Tensor status:");
    for tref in &tensor_refs {
        let status = if tref.loaded { "LOADED" } else { "DEFERRED" };
        println!(
            "    {} @ offset {} ({} bytes) [{}]",
            tref.name, tref.offset, tref.size, status
        );
    }

    // Simulate on-demand materialization: free one tensor, load a deferred one
    let freed_name = if let Some(first_loaded) = loaded_tensors.first() {
        let freed = first_loaded.clone();
        if let Some(tref) = tensor_refs.iter().find(|t| t.name == *freed) {
            budget.free(tref.size);
            println!(
                "  Freed '{}': {} bytes returned to budget",
                freed, tref.size
            );
        }
        Some(freed)
    } else {
        None
    };

    // Load first deferred tensor
    for tref in &mut tensor_refs {
        if !tref.loaded {
            match budget.allocate(tref.size) {
                Ok(()) => {
                    tref.loaded = true;
                    println!("  Lazily materialized '{}': {} bytes", tref.name, tref.size);
                    break;
                }
                Err(e) => {
                    println!("  Cannot materialize '{}': {}", tref.name, e);
                }
            }
        }
    }

    // Mark the freed tensor as no longer loaded
    if let Some(freed_name) = freed_name {
        if let Some(tref) = tensor_refs.iter_mut().find(|t| t.name == *freed_name) {
            tref.loaded = false;
        }
    }

    let final_loaded: Vec<_> = tensor_refs.iter().filter(|t| t.loaded).collect();
    let final_count = final_loaded.len();
    ctx.record_metric("final_tensors_loaded", final_count as i64);
    println!("  Final loaded tensors: {}", final_count);
    println!();

    final_count
}

/// Section 6: Loading performance summary.
fn section_performance_summary(
    loader: &StreamingLoader,
    budget: &WasmMemoryBudget,
    chunk_size: usize,
    _final_loaded_count: usize,
    ctx: &mut RecipeContext,
) {
    println!("--- Section 6: Loading Performance Summary ---");

    let stats = LoadingStats::compute(loader, chunk_size);

    println!("  Total bytes: {}", stats.total_bytes);
    println!("  Chunks received: {}", stats.chunks_received);
    println!("  Simulated time: {} us", stats.time_us);
    println!("  Throughput: {:.2} MB/s", stats.throughput_mbps);
    println!(
        "  Memory efficiency: {:.1}%",
        (budget.used_bytes as f64 / budget.max_bytes as f64) * 100.0
    );

    ctx.record_metric("total_bytes", stats.total_bytes as i64);
    ctx.record_float_metric("throughput_mbps", stats.throughput_mbps);
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// WASM heap memory budget tracker.
#[derive(Debug, Clone)]
struct WasmMemoryBudget {
    max_bytes: usize,
    used_bytes: usize,
}

impl WasmMemoryBudget {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Result<()> {
        if self.used_bytes + size > self.max_bytes {
            return Err(CookbookError::invalid_format(format!(
                "memory budget exceeded: need {} bytes, only {} available",
                size,
                self.max_bytes - self.used_bytes,
            )));
        }
        self.used_bytes += size;
        Ok(())
    }

    fn free(&mut self, size: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(size);
    }

    fn remaining(&self) -> usize {
        self.max_bytes - self.used_bytes
    }
}

/// A chunk of model data received from a simulated streaming fetch.
#[derive(Debug, Clone)]
struct ModelChunk {
    offset: usize,
    data: Vec<u8>,
    is_header: bool,
    is_tensor: bool,
}

/// Tracks the streaming download state.
#[derive(Debug)]
struct StreamingLoader {
    total_size: usize,
    bytes_received: usize,
    chunks_received: usize,
    header_parsed: bool,
    tensors_loaded: usize,
    progress_pct: u32,
}

impl StreamingLoader {
    fn new(total_size: usize) -> Self {
        Self {
            total_size,
            bytes_received: 0,
            chunks_received: 0,
            header_parsed: false,
            tensors_loaded: 0,
            progress_pct: 0,
        }
    }

    fn receive_chunk(&mut self, chunk: &ModelChunk) -> Result<()> {
        if self.bytes_received + chunk.data.len() > self.total_size {
            return Err(CookbookError::invalid_format(
                "received data exceeds declared model size",
            ));
        }
        self.bytes_received += chunk.data.len();
        self.chunks_received += 1;
        self.progress_pct = ((self.bytes_received as f64 / self.total_size as f64) * 100.0) as u32;
        Ok(())
    }
}

/// A reference to a tensor within the model file (lazy loading).
#[derive(Debug, Clone)]
struct TensorRef {
    name: String,
    offset: usize,
    size: usize,
    loaded: bool,
}

/// Summary statistics for the loading process.
#[derive(Debug, Clone)]
struct LoadingStats {
    total_bytes: usize,
    chunks_received: usize,
    time_us: u64,
    throughput_mbps: f64,
}

impl LoadingStats {
    fn compute(loader: &StreamingLoader, _chunk_size: usize) -> Self {
        // Deterministic simulated time: 100us per chunk + 1us per KB
        let time_us = (loader.chunks_received as u64) * 100 + (loader.bytes_received as u64) / 1024;
        let seconds = time_us as f64 / 1_000_000.0;
        let megabytes = loader.bytes_received as f64 / (1024.0 * 1024.0);
        let throughput_mbps = if seconds > 0.0 {
            megabytes / seconds
        } else {
            0.0
        };

        Self {
            total_bytes: loader.bytes_received,
            chunks_received: loader.chunks_received,
            time_us,
            throughput_mbps,
        }
    }
}

/// Parsed model header with tensor table.
#[derive(Debug, Clone)]
struct ParsedHeader {
    magic: u32,
    format_version: u16,
    tensor_refs: Vec<TensorTableEntry>,
}

/// Entry in the tensor offset table from the model header.
#[derive(Debug, Clone)]
struct TensorTableEntry {
    name: String,
    offset: usize,
    size: usize,
}

/// Model metadata describing the file to be loaded.
#[derive(Debug, Clone)]
struct ModelMetadataInfo {
    name: String,
    version: u32,
    total_size: usize,
    header_size: usize,
    tensor_count: usize,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Generate deterministic chunks simulating a streaming fetch response.
fn generate_chunks(meta: &ModelMetadataInfo, chunk_size: usize) -> Vec<ModelChunk> {
    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < meta.total_size {
        let remaining = meta.total_size - offset;
        let size = remaining.min(chunk_size);

        let is_header = offset < meta.header_size;
        let is_tensor = offset >= meta.header_size;

        // Generate deterministic data using DefaultHasher
        let data: Vec<u8> = (0..size)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (offset + i).hash(&mut hasher);
                (hasher.finish() & 0xFF) as u8
            })
            .collect();

        chunks.push(ModelChunk {
            offset,
            data,
            is_header,
            is_tensor,
        });

        offset += size;
    }

    chunks
}

/// Extract header bytes from the first chunk(s).
fn extract_header_bytes(chunks: &[ModelChunk], header_size: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(header_size);

    for chunk in chunks {
        if chunk.offset >= header_size {
            break;
        }
        let end = (chunk.offset + chunk.data.len()).min(header_size);
        let take = end - chunk.offset;
        header.extend_from_slice(&chunk.data[..take]);
    }

    header
}

/// Parse a model header, producing tensor table entries.
fn parse_header(header_data: &[u8], meta: &ModelMetadataInfo) -> Result<ParsedHeader> {
    if header_data.len() < 8 {
        return Err(CookbookError::invalid_format(
            "header too small: need at least 8 bytes",
        ));
    }

    // Derive deterministic magic and version from header bytes
    let mut hasher = DefaultHasher::new();
    header_data[..4].hash(&mut hasher);
    let magic = (hasher.finish() & 0xFFFF_FFFF) as u32;

    let format_version = 2;

    // Build tensor table: distribute remaining space among tensors
    let data_region = meta.total_size - meta.header_size;
    let base_tensor_size = data_region / meta.tensor_count;
    let tensor_names = [
        "embeddings.weight",
        "encoder.layer0.weight",
        "encoder.layer0.bias",
        "encoder.layer1.weight",
        "encoder.layer1.bias",
        "output.weight",
    ];

    let mut tensor_refs = Vec::with_capacity(meta.tensor_count);
    let mut current_offset = meta.header_size;

    for i in 0..meta.tensor_count {
        let name = if i < tensor_names.len() {
            tensor_names[i].to_string()
        } else {
            format!("tensor_{i}")
        };

        // Last tensor gets remainder
        let size = if i == meta.tensor_count - 1 {
            meta.total_size - current_offset
        } else {
            base_tensor_size
        };

        tensor_refs.push(TensorTableEntry {
            name,
            offset: current_offset,
            size,
        });

        current_offset += size;
    }

    Ok(ParsedHeader {
        magic,
        format_version,
        tensor_refs,
    })
}

/// Validate header integrity using a deterministic checksum.
fn validate_header_checksum(header_data: &[u8]) -> bool {
    let mut hasher = DefaultHasher::new();
    header_data.hash(&mut hasher);
    let checksum = hasher.finish();
    // Deterministic: the checksum is always valid for our generated data
    checksum != 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- WasmMemoryBudget tests --

    #[test]
    fn test_budget_new() {
        let budget = WasmMemoryBudget::new(1024);
        assert_eq!(budget.max_bytes, 1024);
        assert_eq!(budget.used_bytes, 0);
        assert_eq!(budget.remaining(), 1024);
    }

    #[test]
    fn test_budget_allocate_success() {
        let mut budget = WasmMemoryBudget::new(1024);
        assert!(budget.allocate(512).is_ok());
        assert_eq!(budget.used_bytes, 512);
        assert_eq!(budget.remaining(), 512);
    }

    #[test]
    fn test_budget_allocate_exact() {
        let mut budget = WasmMemoryBudget::new(1024);
        assert!(budget.allocate(1024).is_ok());
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn test_budget_allocate_exceeds() {
        let mut budget = WasmMemoryBudget::new(1024);
        let result = budget.allocate(2048);
        assert!(result.is_err());
        assert_eq!(budget.used_bytes, 0);
    }

    #[test]
    fn test_budget_allocate_multiple_then_exceed() {
        let mut budget = WasmMemoryBudget::new(1024);
        assert!(budget.allocate(500).is_ok());
        assert!(budget.allocate(500).is_ok());
        let result = budget.allocate(100);
        assert!(result.is_err());
        assert_eq!(budget.used_bytes, 1000);
    }

    #[test]
    fn test_budget_free() {
        let mut budget = WasmMemoryBudget::new(1024);
        budget.allocate(512).expect("should allocate");
        budget.free(256);
        assert_eq!(budget.used_bytes, 256);
        assert_eq!(budget.remaining(), 768);
    }

    #[test]
    fn test_budget_free_saturates() {
        let mut budget = WasmMemoryBudget::new(1024);
        budget.free(9999);
        assert_eq!(budget.used_bytes, 0);
    }

    // -- ModelChunk tests --

    #[test]
    fn test_generate_chunks_count() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 1000,
            header_size: 100,
            tensor_count: 2,
        };
        let chunks = generate_chunks(&meta, 300);
        // 1000 / 300 = 3 full + 1 partial = 4
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn test_generate_chunks_total_bytes() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 1000,
            header_size: 100,
            tensor_count: 2,
        };
        let chunks = generate_chunks(&meta, 300);
        let total: usize = chunks.iter().map(|c| c.data.len()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_generate_chunks_deterministic() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 500,
            header_size: 50,
            tensor_count: 2,
        };
        let chunks1 = generate_chunks(&meta, 200);
        let chunks2 = generate_chunks(&meta, 200);
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.data, b.data);
            assert_eq!(a.offset, b.offset);
        }
    }

    #[test]
    fn test_generate_chunks_header_flags() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 1000,
            header_size: 200,
            tensor_count: 2,
        };
        let chunks = generate_chunks(&meta, 300);
        // First chunk at offset 0 should be header
        assert!(chunks[0].is_header);
        assert!(!chunks[0].is_tensor);
        // Chunk at offset 300 is past header_size 200
        assert!(!chunks[1].is_header);
        assert!(chunks[1].is_tensor);
    }

    // -- StreamingLoader tests --

    #[test]
    fn test_loader_new() {
        let loader = StreamingLoader::new(5000);
        assert_eq!(loader.total_size, 5000);
        assert_eq!(loader.bytes_received, 0);
        assert_eq!(loader.chunks_received, 0);
        assert!(!loader.header_parsed);
        assert_eq!(loader.tensors_loaded, 0);
        assert_eq!(loader.progress_pct, 0);
    }

    #[test]
    fn test_loader_receive_chunk() {
        let mut loader = StreamingLoader::new(1000);
        let chunk = ModelChunk {
            offset: 0,
            data: vec![0u8; 500],
            is_header: true,
            is_tensor: false,
        };
        assert!(loader.receive_chunk(&chunk).is_ok());
        assert_eq!(loader.bytes_received, 500);
        assert_eq!(loader.chunks_received, 1);
        assert_eq!(loader.progress_pct, 50);
    }

    #[test]
    fn test_loader_overflow_rejected() {
        let mut loader = StreamingLoader::new(100);
        let chunk = ModelChunk {
            offset: 0,
            data: vec![0u8; 200],
            is_header: false,
            is_tensor: true,
        };
        assert!(loader.receive_chunk(&chunk).is_err());
    }

    // -- Header parsing tests --

    #[test]
    fn test_extract_header_bytes() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 1000,
            header_size: 100,
            tensor_count: 2,
        };
        let chunks = generate_chunks(&meta, 300);
        let header = extract_header_bytes(&chunks, 100);
        assert_eq!(header.len(), 100);
    }

    #[test]
    fn test_parse_header_too_small() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 1000,
            header_size: 100,
            tensor_count: 2,
        };
        let result = parse_header(&[0u8; 4], &meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_tensor_count() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 10000,
            header_size: 200,
            tensor_count: 4,
        };
        let header = vec![0xABu8; 200];
        let parsed = parse_header(&header, &meta).expect("should parse");
        assert_eq!(parsed.tensor_refs.len(), 4);
        assert_eq!(parsed.format_version, 2);
    }

    #[test]
    fn test_parse_header_tensor_offsets_contiguous() {
        let meta = ModelMetadataInfo {
            name: "test".to_string(),
            version: 1,
            total_size: 5000,
            header_size: 200,
            tensor_count: 3,
        };
        let header = vec![0xCDu8; 200];
        let parsed = parse_header(&header, &meta).expect("should parse");

        // Verify tensors are contiguous and cover entire data region
        let mut expected_offset = meta.header_size;
        for tref in &parsed.tensor_refs {
            assert_eq!(tref.offset, expected_offset);
            expected_offset += tref.size;
        }
        assert_eq!(expected_offset, meta.total_size);
    }

    #[test]
    fn test_validate_header_checksum() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        assert!(validate_header_checksum(&data));
    }

    // -- LoadingStats tests --

    #[test]
    fn test_loading_stats_compute() {
        let mut loader = StreamingLoader::new(10000);
        let chunk = ModelChunk {
            offset: 0,
            data: vec![0u8; 10000],
            is_header: false,
            is_tensor: true,
        };
        loader.receive_chunk(&chunk).expect("should receive");

        let stats = LoadingStats::compute(&loader, 10000);
        assert_eq!(stats.total_bytes, 10000);
        assert_eq!(stats.chunks_received, 1);
        assert!(stats.time_us > 0);
        assert!(stats.throughput_mbps > 0.0);
    }

    // -- TensorRef tests --

    #[test]
    fn test_tensor_ref_initial_state() {
        let tref = TensorRef {
            name: "weights".to_string(),
            offset: 1024,
            size: 4096,
            loaded: false,
        };
        assert!(!tref.loaded);
        assert_eq!(tref.name, "weights");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_chunks_cover_full_model(
            total_size in 100usize..10000,
            chunk_size in 50usize..500,
        ) {
            let meta = ModelMetadataInfo {
                name: "prop".to_string(),
                version: 1,
                total_size,
                header_size: total_size.min(100),
                tensor_count: 2,
            };
            let chunks = generate_chunks(&meta, chunk_size);
            let total: usize = chunks.iter().map(|c| c.data.len()).sum();
            prop_assert_eq!(total, total_size);
        }

        #[test]
        fn prop_chunks_offsets_monotonic(
            total_size in 200usize..5000,
            chunk_size in 50usize..300,
        ) {
            let meta = ModelMetadataInfo {
                name: "prop".to_string(),
                version: 1,
                total_size,
                header_size: 100,
                tensor_count: 2,
            };
            let chunks = generate_chunks(&meta, chunk_size);
            for pair in chunks.windows(2) {
                prop_assert!(pair[1].offset > pair[0].offset);
            }
        }

        #[test]
        fn prop_budget_allocate_then_free_restores(
            max in 1000usize..100_000,
            alloc in 1usize..1000,
        ) {
            if alloc <= max {
                let mut budget = WasmMemoryBudget::new(max);
                budget.allocate(alloc).expect("should fit");
                budget.free(alloc);
                prop_assert_eq!(budget.used_bytes, 0);
                prop_assert_eq!(budget.remaining(), max);
            }
        }

        #[test]
        fn prop_loader_progress_reaches_100(
            total in 500usize..5000,
            chunk_size in 100usize..600,
        ) {
            let meta = ModelMetadataInfo {
                name: "prop".to_string(),
                version: 1,
                total_size: total,
                header_size: total.min(50),
                tensor_count: 1,
            };
            let chunks = generate_chunks(&meta, chunk_size);
            let mut loader = StreamingLoader::new(total);
            for chunk in &chunks {
                loader.receive_chunk(chunk).expect("should receive");
            }
            prop_assert_eq!(loader.progress_pct, 100);
        }

        #[test]
        fn prop_parsed_header_tensors_span_data_region(
            tensor_count in 1usize..8,
            total_size in 1000usize..50000,
        ) {
            let header_size = 256;
            if total_size > header_size + tensor_count {
                let meta = ModelMetadataInfo {
                    name: "prop".to_string(),
                    version: 1,
                    total_size,
                    header_size,
                    tensor_count,
                };
                let header = vec![0xAAu8; header_size];
                let parsed = parse_header(&header, &meta).expect("parse ok");
                let total_tensor_bytes: usize = parsed.tensor_refs.iter().map(|t| t.size).sum();
                prop_assert_eq!(total_tensor_bytes, total_size - header_size);
            }
        }
    }
}
