#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Section helpers — one per logical block of main()
// ---------------------------------------------------------------------------

/// Section 1: Define WASM memory constraints and model metadata.
pub fn section_memory_constraints(
    ctx: &mut RecipeContext,
) -> (WasmMemoryBudget, ModelMetadataInfo) {
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
pub fn section_chunked_download(
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
pub fn section_header_parsing(
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
pub fn section_budgeted_tensor_loading(
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
pub fn section_lazy_materialization(
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
pub fn section_performance_summary(
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
