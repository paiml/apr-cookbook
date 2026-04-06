#![allow(unused_imports)]
//! # Recipe: Memory-Mapped Inference
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
//! **Category**: Acceleration - Memory Optimization
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Demonstrates memory-mapped model loading vs eager loading. Memory-mapped
//! access provides near-instant file open, demand-paged reads, and reduced
//! resident memory when only a subset of tensors is accessed during inference.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_mmap_inference
//! ```
//!
//! ## Toyota Way Principles
//! - **Muda** (Waste elimination): Only load the pages you actually need
//! - **Jidoka** (Quality built-in): Track page faults to verify demand paging
//! - **Genchi Genbutsu** (Go and see): Measure actual RSS, not theoretical max
//!
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

use apr_cookbook::prelude::*;
use rand::Rng;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!("========================================================");
    println!("  Memory-Mapped Inference");
    println!("  Eager loading vs mmap: load time, peak RSS, latency");
    println!("========================================================");
    println!();

    let mut ctx = RecipeContext::new("acceleration_mmap_inference")?;

    // Step 1: Create synthetic model file
    let tensors = create_synthetic_model(ctx.rng());
    let model_path = ctx.path("model.bin");
    write_model_file(&model_path, &tensors)?;

    let file_size = std::fs::metadata(&model_path)?.len() as usize;
    println!("1. Synthetic model written");
    println!("   Path:    {}", model_path.display());
    println!("   Tensors: {NUM_TENSORS} x {ELEMENTS_PER_TENSOR} f64 elements");
    println!(
        "   Size:    {} bytes ({:.1} KB)",
        file_size,
        file_size as f64 / 1024.0
    );
    println!();

    // Step 2: Eager loading
    let (eager_metrics, eager_result) = run_eager_inference(&model_path, &tensors)?;
    println!("2. Eager loading complete");
    print_metrics(&eager_metrics);

    // Step 3: Memory-mapped loading
    let (mmap_metrics, mmap_result, page_report) = run_mmap_inference(&model_path, &tensors)?;
    println!("3. Memory-mapped loading complete");
    print_metrics(&mmap_metrics);

    // Step 4: Comparison table
    print_comparison(&eager_metrics, &eager_result, &mmap_metrics, &mmap_result);

    // Step 5: Page access pattern
    print_page_access(&page_report);

    // Record metrics
    ctx.record_float_metric("eager_load_ms", eager_metrics.load_time_ms);
    ctx.record_float_metric("mmap_load_ms", mmap_metrics.load_time_ms);
    ctx.record_metric("eager_peak_bytes", eager_metrics.peak_memory_bytes as i64);
    ctx.record_metric("mmap_peak_bytes", mmap_metrics.peak_memory_bytes as i64);
    ctx.record_float_metric("eager_inference_ms", eager_result.latency_ms);
    ctx.record_float_metric("mmap_inference_ms", mmap_result.latency_ms);

    println!("\nMemory-mapped inference example complete.");
    Ok(())
}

// ============================================================================
// Section 1: Synthetic Model Creation
// ============================================================================

/// Generate `NUM_TENSORS` tensors with deterministic random weights.
fn create_synthetic_model(rng: &mut impl Rng) -> Vec<TensorRecord> {
    let names = [
        "layer0.weight",
        "layer1.weight",
        "layer2.weight",
        "layer3.weight",
    ];
    names
        .iter()
        .map(|&name| {
            let data: Vec<f64> = (0..ELEMENTS_PER_TENSOR)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            TensorRecord {
                name: name.to_string(),
                data,
            }
        })
        .collect()
}

/// Serialize tensors to a flat binary file (little-endian f64 values).
fn write_model_file(path: &std::path::Path, tensors: &[TensorRecord]) -> Result<()> {
    let mut bytes = Vec::with_capacity(NUM_TENSORS * ELEMENTS_PER_TENSOR * 8);
    for tensor in tensors {
        for &val in &tensor.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
    }
    std::fs::write(path, &bytes)?;
    Ok(())
}

/// Read raw bytes back into tensor records.
fn parse_tensors_from_bytes(raw: &[u8], names: &[&str]) -> Vec<TensorRecord> {
    let bytes_per_tensor = ELEMENTS_PER_TENSOR * 8;
    names
        .iter()
        .enumerate()
        .map(|(i, &name)| {
            let start = i * bytes_per_tensor;
            let end = start + bytes_per_tensor;
            let slice = &raw[start..end];
            let data: Vec<f64> = slice
                .chunks_exact(8)
                .map(|chunk| {
                    let arr: [u8; 8] = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    f64::from_le_bytes(arr)
                })
                .collect();
            TensorRecord {
                name: name.to_string(),
                data,
            }
        })
        .collect()
}

// ============================================================================
// Section 2: Eager Loading
// ============================================================================

/// Load the entire model file into memory, parse tensors, run inference.
fn run_eager_inference(
    path: &std::path::Path,
    _reference: &[TensorRecord],
) -> Result<(LoadMetrics, InferenceResult)> {
    let names = [
        "layer0.weight",
        "layer1.weight",
        "layer2.weight",
        "layer3.weight",
    ];
    let total_pages = {
        let file_len = std::fs::metadata(path)?.len() as usize;
        file_len.div_ceil(PAGE_SIZE)
    };

    // Eager: read the entire file at once
    let load_start = Instant::now();
    let raw = std::fs::read(path)?;
    let tensors = parse_tensors_from_bytes(&raw, &names);
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let peak_memory_bytes = raw.len();

    // Run forward pass (only uses layers 0..ACTIVE_LAYERS)
    let infer_start = Instant::now();
    let output = forward_pass(&tensors);
    let latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let metrics = LoadMetrics {
        strategy: LoadStrategy::Eager,
        load_time_ms,
        peak_memory_bytes,
        pages_resident: total_pages, // eager loads everything
        pages_total: total_pages,
    };

    let result = InferenceResult {
        strategy: LoadStrategy::Eager,
        output,
        latency_ms,
    };

    Ok((metrics, result))
}

// ============================================================================
// Section 3: Memory-Mapped Loading
// ============================================================================

/// Open the model file as a simulated mmap, run inference, report page faults.
fn run_mmap_inference(
    path: &std::path::Path,
    reference_tensors: &[TensorRecord],
) -> Result<(LoadMetrics, InferenceResult, Vec<PageAccess>)> {
    let raw = std::fs::read(path)?;
    let total_size = raw.len();

    // Mmap open is near-instant: we just create the view without reading data
    let load_start = Instant::now();
    let mut view = MmapView::new(raw);
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // Run forward pass, accessing only layers 0..ACTIVE_LAYERS via the view
    let infer_start = Instant::now();
    let output = forward_pass_mmap(&mut view);
    let latency_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let resident = view.resident_pages();
    let page_report = view.page_report(reference_tensors);

    // Peak memory = only the pages that were actually faulted in
    let peak_memory_bytes = resident * PAGE_SIZE;

    let metrics = LoadMetrics {
        strategy: LoadStrategy::MemoryMapped,
        load_time_ms,
        peak_memory_bytes: peak_memory_bytes.min(total_size),
        pages_resident: resident,
        pages_total: view.page_count,
    };

    let result = InferenceResult {
        strategy: LoadStrategy::MemoryMapped,
        output,
        latency_ms,
    };

    Ok((metrics, result, page_report))
}

// ============================================================================
// Section 4: Forward Pass
// ============================================================================

/// Simple forward pass: accumulate weighted sums from layers 0..`ACTIVE_LAYERS`.
///
/// This intentionally skips the last tensor to demonstrate that mmap only
/// pages in the data that is actually accessed.
fn forward_pass(tensors: &[TensorRecord]) -> Vec<f64> {
    let output_size = 16;
    let mut output = vec![0.0_f64; output_size];

    for tensor in tensors.iter().take(ACTIVE_LAYERS) {
        for (j, out) in output.iter_mut().enumerate() {
            let stride = tensor.data.len() / output_size;
            let start = j * stride;
            let end = start + stride;
            let sum: f64 = tensor.data[start..end].iter().sum();
            *out += sum / stride as f64;
        }
    }

    // Apply tanh activation
    for val in &mut output {
        *val = val.tanh();
    }

    output
}

/// Forward pass through the mmap view, reading only the needed byte ranges.
fn forward_pass_mmap(view: &mut MmapView) -> Vec<f64> {
    let output_size = 16;
    let bytes_per_tensor = ELEMENTS_PER_TENSOR * 8;
    let mut output = vec![0.0_f64; output_size];

    for layer in 0..ACTIVE_LAYERS {
        let tensor_offset = layer * bytes_per_tensor;
        let raw = view.read_range(tensor_offset, bytes_per_tensor);
        let elements: Vec<f64> = raw
            .chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ];
                f64::from_le_bytes(arr)
            })
            .collect();

        for (j, out) in output.iter_mut().enumerate() {
            let stride = elements.len() / output_size;
            let start = j * stride;
            let end = start + stride;
            let sum: f64 = elements[start..end].iter().sum();
            *out += sum / stride as f64;
        }
    }

    for val in &mut output {
        *val = val.tanh();
    }

    output
}

// ============================================================================
// Section 5: Reporting
// ============================================================================

/// Print metrics for a single loading strategy.
fn print_metrics(m: &LoadMetrics) {
    println!("   Strategy:     {}", m.strategy);
    println!("   Load time:    {:.3} ms", m.load_time_ms);
    println!(
        "   Peak memory:  {} bytes ({:.1} KB)",
        m.peak_memory_bytes,
        m.peak_memory_bytes as f64 / 1024.0
    );
    println!(
        "   Pages:        {}/{} resident",
        m.pages_resident, m.pages_total
    );
    println!();
}

/// Print side-by-side comparison table.
fn print_comparison(
    eager_m: &LoadMetrics,
    eager_r: &InferenceResult,
    mmap_m: &LoadMetrics,
    mmap_r: &InferenceResult,
) {
    println!("4. Comparison");
    println!("   ──────────────────────────────────────────────────────────");
    println!("   {:>20} {:>16} {:>16}", "Metric", "Eager", "MemoryMapped");
    println!("   ──────────────────────────────────────────────────────────");
    println!(
        "   {:>20} {:>15.3}  {:>15.3}",
        "load_time_ms", eager_m.load_time_ms, mmap_m.load_time_ms
    );
    println!(
        "   {:>20} {:>16} {:>16}",
        "peak_memory_bytes", eager_m.peak_memory_bytes, mmap_m.peak_memory_bytes
    );
    println!(
        "   {:>20} {:>15.3}  {:>15.3}",
        "inference_ms", eager_r.latency_ms, mmap_r.latency_ms
    );
    println!(
        "   {:>20} {:>16} {:>16}",
        "pages_resident",
        format!("{}/{}", eager_m.pages_resident, eager_m.pages_total),
        format!("{}/{}", mmap_m.pages_resident, mmap_m.pages_total),
    );
    println!("   ──────────────────────────────────────────────────────────");

    let memory_saving = if eager_m.peak_memory_bytes > 0 {
        (1.0 - mmap_m.peak_memory_bytes as f64 / eager_m.peak_memory_bytes as f64) * 100.0
    } else {
        0.0
    };
    println!("   Memory saving (mmap vs eager): {:.1}%", memory_saving);
    println!();
}

/// Print the page access pattern showing which pages were touched.
fn print_page_access(pages: &[PageAccess]) {
    println!("5. Page Access Pattern");
    println!("   ──────────────────────────────────────────────────────────");
    println!(
        "   {:>6} {:>10} {:>6} {:>9} {:>18}",
        "PageID", "Offset", "Size", "Accessed", "Tensor"
    );
    println!("   ──────────────────────────────────────────────────────────");

    for page in pages {
        let mark = if page.accessed { "YES" } else { "---" };
        println!(
            "   {:>6} {:>10} {:>6} {:>9} {:>18}",
            page.page_id, page.offset, page.size, mark, page.tensor_name
        );
    }

    let accessed_count = pages.iter().filter(|p| p.accessed).count();
    let total = pages.len();
    println!("   ──────────────────────────────────────────────────────────");
    println!(
        "   Pages accessed: {}/{} ({:.1}%)",
        accessed_count,
        total,
        accessed_count as f64 / total as f64 * 100.0
    );
    println!();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a deterministic recipe context for tests.
    fn test_ctx() -> RecipeContext {
        RecipeContext::new("test_mmap_inference").expect("context creation")
    }

    #[test]
    fn test_create_synthetic_model_produces_correct_count() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        assert_eq!(tensors.len(), NUM_TENSORS);
    }

    #[test]
    fn test_each_tensor_has_correct_element_count() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        for t in &tensors {
            assert_eq!(t.data.len(), ELEMENTS_PER_TENSOR);
        }
    }

    #[test]
    fn test_write_and_read_roundtrip() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let path = ctx.path("roundtrip.bin");
        write_model_file(&path, &tensors).expect("write");

        let raw = std::fs::read(&path).expect("read");
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        let parsed = parse_tensors_from_bytes(&raw, &names);

        for (orig, parsed) in tensors.iter().zip(parsed.iter()) {
            assert_eq!(orig.name, parsed.name);
            assert_eq!(orig.data.len(), parsed.data.len());
            for (a, b) in orig.data.iter().zip(parsed.data.iter()) {
                assert!((a - b).abs() < 1e-15, "mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_forward_pass_output_size() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let output = forward_pass(&tensors);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_forward_pass_values_bounded() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let output = forward_pass(&tensors);
        // tanh output is always in (-1, 1)
        for &v in &output {
            assert!(v.abs() < 1.0, "tanh output must be in (-1,1), got {v}");
        }
    }

    #[test]
    fn test_mmap_view_tracks_page_access() {
        let data = vec![0u8; PAGE_SIZE * 4];
        let mut view = MmapView::new(data);
        assert_eq!(view.resident_pages(), 0);

        // Access page 1 only
        let _ = view.read_range(PAGE_SIZE, 8);
        assert_eq!(view.resident_pages(), 1);
        assert!(!view.page_accessed[0]);
        assert!(view.page_accessed[1]);
    }

    #[test]
    fn test_mmap_forward_pass_matches_eager() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let path = ctx.path("match.bin");
        write_model_file(&path, &tensors).expect("write");

        let raw = std::fs::read(&path).expect("read");
        let mut view = MmapView::new(raw);

        let eager_output = forward_pass(&tensors);
        let mmap_output = forward_pass_mmap(&mut view);

        assert_eq!(eager_output.len(), mmap_output.len());
        for (a, b) in eager_output.iter().zip(mmap_output.iter()) {
            assert!((a - b).abs() < 1e-10, "eager vs mmap mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_mmap_skips_unused_tensor_pages() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let path = ctx.path("skip.bin");
        write_model_file(&path, &tensors).expect("write");

        let raw = std::fs::read(&path).expect("read");
        let mut view = MmapView::new(raw);
        let _ = forward_pass_mmap(&mut view);

        // Tensor 3 (the last one) should NOT have any pages accessed
        let bytes_per_tensor = ELEMENTS_PER_TENSOR * 8;
        let tensor3_first_page = (3 * bytes_per_tensor) / PAGE_SIZE;
        for p in tensor3_first_page..view.page_count {
            assert!(
                !view.page_accessed[p],
                "page {p} in tensor 3 should not be accessed"
            );
        }
    }

    #[test]
    fn test_load_strategy_display() {
        assert_eq!(format!("{}", LoadStrategy::Eager), "Eager");
        assert_eq!(format!("{}", LoadStrategy::MemoryMapped), "MemoryMapped");
    }

    #[test]
    fn test_page_report_annotations() {
        let mut ctx = test_ctx();
        let tensors = create_synthetic_model(ctx.rng());
        let path = ctx.path("report.bin");
        write_model_file(&path, &tensors).expect("write");

        let raw = std::fs::read(&path).expect("read");
        let mut view = MmapView::new(raw);
        let _ = forward_pass_mmap(&mut view);
        let report = view.page_report(&tensors);

        assert!(!report.is_empty());
        // First page should belong to layer0.weight
        assert_eq!(report[0].tensor_name, "layer0.weight");
        // Every page should have a non-empty tensor name
        for page in &report {
            assert!(!page.tensor_name.is_empty());
        }
    }
}
