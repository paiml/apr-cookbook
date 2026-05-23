#![allow(unused_imports)]
//! # Recipe: WASM Model Loader Pipeline
//!
//! Contract: contracts/recipe-iiur-v1.yaml
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
