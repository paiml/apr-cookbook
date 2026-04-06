#![allow(unused_imports)]
//! # Recipe: Memory-Mapped Lazy Model Loading
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/mmap-inference-v1.yaml
//! **Category**: Inference Patterns
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: std only (simulated mmap via seek/read)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A - file I/O)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate memory-mapped lazy loading for models approaching RAM limits.
//! On a 16 GB machine, mmap + lazy loading lets you work with models larger
//! than free RAM by reading only the tensor slices you actually need.
//!
//! ## Toyota Way: ムダ (Muda) - Waste Elimination
//! Never load bytes you will not use. Lazy loading eliminates the waste of
//! reading an entire multi-GB model when inference touches a fraction of it.
//!
//! ## Run Command
//! ```bash
//! cargo run --example inference_mmap_lazy_load
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
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use apr_cookbook::prelude::*;
use rand::Rng;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_mmap_lazy_load")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Memory-mapped lazy loading for large-model inference");
    println!();

    // =========================================================================
    // Section 1: Create Synthetic Model Files
    // =========================================================================
    println!("1. Creating Synthetic Model Files");
    println!("   -----------------------------------------------");

    let sizes_mb: &[usize] = &[10, 50, 100];
    let mut model_paths = Vec::new();
    let mut model_tensors = Vec::new();

    for &size in sizes_mb {
        let path = ctx.path(&format!("model_{size}mb.bin"));
        let rng = ctx.rng();
        let tensors = create_test_model(&path, size, rng)?;
        println!(
            "   Created {}MB model: {} tensors, file = {} bytes",
            size,
            tensors.len(),
            std::fs::metadata(&path)?.len()
        );
        model_paths.push(path);
        model_tensors.push(tensors);
    }
    println!();

    ctx.record_metric("model_variants", sizes_mb.len() as i64);

    // =========================================================================
    // Section 2: Eager Load Benchmark
    // =========================================================================
    println!("2. Eager Load Benchmark (full file read)");
    println!("   -----------------------------------------------");
    println!("   {:>8} {:>12} {:>14}", "Size", "Time", "Bytes Read");
    println!("   {}", "-".repeat(36));

    let mut eager_durations = Vec::new();
    for (i, path) in model_paths.iter().enumerate() {
        let (dur, bytes) = benchmark_eager_load(path)?;
        println!(
            "   {:>6}MB {:>10.2}ms {:>12} B",
            sizes_mb[i],
            dur.as_secs_f64() * 1000.0,
            bytes
        );
        eager_durations.push(dur);
    }
    println!();

    // =========================================================================
    // Section 3: Lazy / Mmap Load Benchmark (20% of tensors)
    // =========================================================================
    println!("3. Lazy Load Benchmark (20% of tensors)");
    println!("   -----------------------------------------------");
    println!(
        "   {:>8} {:>12} {:>14} {:>10}",
        "Size", "Time", "Bytes Read", "Fraction"
    );
    println!("   {}", "-".repeat(48));

    let mut lazy_durations = Vec::new();
    for (i, (path, tensors)) in model_paths.iter().zip(model_tensors.iter()).enumerate() {
        let n_to_load = (tensors.len() / 5).max(1);
        let indices: Vec<usize> = (0..n_to_load).collect();
        let (dur, bytes) = benchmark_lazy_load(path, tensors, &indices)?;
        let file_size = std::fs::metadata(path)?.len();
        let fraction = bytes as f64 / file_size as f64 * 100.0;
        println!(
            "   {:>6}MB {:>10.2}ms {:>12} B {:>8.1}%",
            sizes_mb[i],
            dur.as_secs_f64() * 1000.0,
            bytes,
            fraction
        );
        lazy_durations.push(dur);
    }
    println!();

    // =========================================================================
    // Section 4: Memory Analysis
    // =========================================================================
    println!("4. Memory Analysis");
    println!("   -----------------------------------------------");

    let largest_idx = sizes_mb.len() - 1;
    let largest_tensors = &model_tensors[largest_idx];
    let largest_path = &model_paths[largest_idx];
    let file_size = std::fs::metadata(largest_path)?.len();
    let n_loaded = (largest_tensors.len() / 5).max(1);
    let bytes_loaded: u64 = largest_tensors
        .iter()
        .take(n_loaded)
        .map(|t| t.length)
        .sum();

    println!(
        "   Model file size:       {} bytes ({} MB)",
        file_size,
        file_size / (1024 * 1024)
    );
    println!("   Total tensors:         {}", largest_tensors.len());
    println!("   Tensors loaded (lazy): {}", n_loaded);
    println!(
        "   Bytes loaded (lazy):   {} bytes ({:.1} MB)",
        bytes_loaded,
        bytes_loaded as f64 / (1024.0 * 1024.0)
    );
    println!(
        "   Memory saved:          {:.1}%",
        (1.0 - bytes_loaded as f64 / file_size as f64) * 100.0
    );
    println!();

    let ram_gb = 16_u64;
    let ram_bytes = ram_gb * 1024 * 1024 * 1024;
    // Hypothetical: a 12 GB model on a 16 GB machine
    let hypothetical_model_gb = 12_u64;
    let hypothetical_fraction = 0.20;
    let hypothetical_loaded = (hypothetical_model_gb as f64 * hypothetical_fraction) as u64;
    println!("   === 16 GB Machine Scenario ===");
    println!("   Model size:          {} GB", hypothetical_model_gb);
    println!("   System RAM:          {} GB", ram_gb);
    println!(
        "   Eager load:          Would consume {:.0}% of RAM",
        hypothetical_model_gb as f64 / ram_gb as f64 * 100.0
    );
    println!(
        "   Lazy load (20%):     ~{} GB resident ({:.0}% of RAM)",
        hypothetical_loaded,
        hypothetical_loaded as f64 / ram_bytes as f64 * 1024.0 * 1024.0 * 1024.0 * 100.0
    );
    println!();

    ctx.record_metric("file_size_bytes", file_size as i64);
    ctx.record_metric("lazy_bytes_loaded", bytes_loaded as i64);

    // =========================================================================
    // Section 5: Latency Comparison Table
    // =========================================================================
    println!("5. Latency Comparison");
    println!("   -----------------------------------------------");
    println!(
        "   {:>8} {:>14} {:>14} {:>10}",
        "Size", "Eager(ms)", "Lazy(ms)", "Speedup"
    );
    println!("   {}", "-".repeat(50));

    for i in 0..sizes_mb.len() {
        let eager_ms = eager_durations[i].as_secs_f64() * 1000.0;
        let lazy_ms = lazy_durations[i].as_secs_f64() * 1000.0;
        let speedup = if lazy_ms > 0.0 {
            eager_ms / lazy_ms
        } else {
            f64::INFINITY
        };
        println!(
            "   {:>6}MB {:>12.2} {:>12.2} {:>8.1}x",
            sizes_mb[i], eager_ms, lazy_ms, speedup
        );
    }
    println!();

    // =========================================================================
    // Section 6: Record Metrics and Report
    // =========================================================================
    println!("6. Metrics Summary");
    println!("   -----------------------------------------------");

    if let Some(eager_dur) = eager_durations.last() {
        ctx.record_duration("eager_load_100mb", *eager_dur);
    }
    if let Some(lazy_dur) = lazy_durations.last() {
        ctx.record_duration("lazy_load_100mb_20pct", *lazy_dur);
    }

    let memory_saving_pct = (1.0 - bytes_loaded as f64 / file_size as f64) * 100.0;
    ctx.record_float_metric("memory_saving_pct", memory_saving_pct);
    ctx.record_metric("tensors_in_largest_model", largest_tensors.len() as i64);

    ctx.report()?;
    println!();
    println!("=== Recipe Complete ===");

    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_model_produces_valid_file() {
        let ctx = RecipeContext::new("test_create_model").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");
        assert!(!tensors.is_empty());
        assert!(path.exists());
    }

    #[test]
    fn test_create_model_file_size() {
        let ctx = RecipeContext::new("test_file_size").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let _tensors = create_test_model(&path, 1, &mut rng).expect("create");
        let meta = std::fs::metadata(&path).expect("meta");
        // 1 MB of data plus header + tensor metadata
        assert!(meta.len() >= 1024 * 1024);
    }

    #[test]
    fn test_simulate_mmap_load_validates_tensors() {
        let ctx = RecipeContext::new("test_mmap_validate").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");
        let loader = simulate_mmap_load(&path, &tensors).expect("mmap");
        assert!(loader.file_size >= 1024 * 1024);
        assert!(loader.mapped_regions.is_empty());
    }

    #[test]
    fn test_lazy_load_single_tensor() {
        let ctx = RecipeContext::new("test_lazy_single").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");
        let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");

        let data = lazy_load_tensor(&mut loader, &tensors[0]).expect("load");
        assert_eq!(data.len(), 65_536); // 256 KiB / 4 bytes per float
        assert_eq!(loader.mapped_regions.len(), 1);
        assert_eq!(loader.total_bytes_read, tensors[0].length);
    }

    #[test]
    fn test_lazy_load_tracks_access_count() {
        let ctx = RecipeContext::new("test_access_count").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");
        let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");

        let _ = lazy_load_tensor(&mut loader, &tensors[0]).expect("load1");
        let _ = lazy_load_tensor(&mut loader, &tensors[0]).expect("load2");

        assert_eq!(loader.mapped_regions.len(), 1);
        assert_eq!(loader.mapped_regions[0].access_count, 2);
        // total_bytes_read only increments on first access
        assert_eq!(loader.total_bytes_read, tensors[0].length);
    }

    #[test]
    fn test_eager_load_reads_entire_file() {
        let ctx = RecipeContext::new("test_eager").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let _tensors = create_test_model(&path, 1, &mut rng).expect("create");

        let (dur, bytes) = benchmark_eager_load(&path).expect("eager");
        let file_len = std::fs::metadata(&path).expect("meta").len() as usize;
        assert_eq!(bytes, file_len);
        assert!(dur.as_nanos() > 0);
    }

    #[test]
    fn test_lazy_load_reads_less_than_eager() {
        let ctx = RecipeContext::new("test_lazy_less").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 2, &mut rng).expect("create");

        let (_, eager_bytes) = benchmark_eager_load(&path).expect("eager");

        // Load 20% of tensors
        let n = (tensors.len() / 5).max(1);
        let indices: Vec<usize> = (0..n).collect();
        let (_, lazy_bytes) = benchmark_lazy_load(&path, &tensors, &indices).expect("lazy");

        assert!(
            lazy_bytes < eager_bytes,
            "lazy {} should be < eager {}",
            lazy_bytes,
            eager_bytes
        );
    }

    #[test]
    fn test_tensor_data_deterministic() {
        let ctx = RecipeContext::new("test_deterministic").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(99);
        let tensors = create_test_model(&path, 1, &mut rng).expect("create");

        let mut loader1 = simulate_mmap_load(&path, &tensors).expect("mmap1");
        let data1 = lazy_load_tensor(&mut loader1, &tensors[0]).expect("load1");

        let mut loader2 = simulate_mmap_load(&path, &tensors).expect("mmap2");
        let data2 = lazy_load_tensor(&mut loader2, &tensors[0]).expect("load2");

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_invalid_model_size_rejected() {
        let ctx = RecipeContext::new("test_invalid_size").expect("ctx");
        let path = ctx.path("tiny.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        // 0 MB should fail
        let result = create_test_model(&path, 0, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_rejects_truncated_file() {
        let ctx = RecipeContext::new("test_truncated").expect("ctx");
        let path = ctx.path("truncated.bin");

        // Write a tiny file
        std::fs::write(&path, b"APRM").expect("write");

        let fake_tensor = ModelTensor {
            name: "fake".to_string(),
            shape: vec![4, 4],
            dtype: "f32".to_string(),
            offset: 1000,
            length: 4096,
        };
        let result = simulate_mmap_load(&path, &[fake_tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_tensors_independent() {
        let ctx = RecipeContext::new("test_multi_tensor").expect("ctx");
        let path = ctx.path("model.bin");
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let tensors = create_test_model(&path, 2, &mut rng).expect("create");
        assert!(tensors.len() >= 2);

        let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");
        let d0 = lazy_load_tensor(&mut loader, &tensors[0]).expect("t0");
        let d1 = lazy_load_tensor(&mut loader, &tensors[1]).expect("t1");

        // Different tensors should have different data (random fill)
        assert_ne!(d0, d1);
        assert_eq!(loader.mapped_regions.len(), 2);
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_lazy_loads_less_bytes(fraction in 0.01f64..0.99) {
            let ctx = RecipeContext::new("prop_lazy_less").expect("ctx");
            let path = ctx.path("model.bin");
            let mut rng = rand::SeedableRng::seed_from_u64(77);
            let tensors = create_test_model(&path, 2, &mut rng).expect("create");

            let n_to_load = ((tensors.len() as f64 * fraction) as usize).max(1).min(tensors.len());
            let indices: Vec<usize> = (0..n_to_load).collect();

            let (_, eager_bytes) = benchmark_eager_load(&path).expect("eager");
            let (_, lazy_bytes) = benchmark_lazy_load(&path, &tensors, &indices).expect("lazy");

            // Lazy should read at most the tensor data we asked for, which is
            // always less than the full file as long as we skip at least one tensor.
            if n_to_load < tensors.len() {
                prop_assert!(
                    lazy_bytes < eager_bytes,
                    "lazy={} should < eager={} at fraction={:.2}",
                    lazy_bytes, eager_bytes, fraction
                );
            }
        }

        #[test]
        fn prop_tensor_roundtrip(seed in 0u64..10_000) {
            let ctx = RecipeContext::new("prop_roundtrip").expect("ctx");
            let path = ctx.path("model.bin");
            let mut rng = rand::SeedableRng::seed_from_u64(seed);
            let tensors = create_test_model(&path, 1, &mut rng).expect("create");

            // Load first tensor twice -- must produce identical f32 data
            let mut loader = simulate_mmap_load(&path, &tensors).expect("mmap");
            let data_a = lazy_load_tensor(&mut loader, &tensors[0]).expect("a");
            let data_b = lazy_load_tensor(&mut loader, &tensors[0]).expect("b");

            prop_assert_eq!(data_a.len(), data_b.len());
            for (i, (a, b)) in data_a.iter().zip(data_b.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < f32::EPSILON,
                    "mismatch at index {}: {} vs {}", i, a, b
                );
            }
        }

        #[test]
        fn prop_deterministic_loads(seed in 0u64..10_000) {
            // Two independent create + load cycles with the same seed
            // must yield identical tensor content.
            let ctx1 = RecipeContext::new("prop_det_1").expect("ctx1");
            let path1 = ctx1.path("model.bin");
            let mut rng1 = rand::SeedableRng::seed_from_u64(seed);
            let tensors1 = create_test_model(&path1, 1, &mut rng1).expect("c1");

            let ctx2 = RecipeContext::new("prop_det_2").expect("ctx2");
            let path2 = ctx2.path("model.bin");
            let mut rng2 = rand::SeedableRng::seed_from_u64(seed);
            let tensors2 = create_test_model(&path2, 1, &mut rng2).expect("c2");

            let mut loader1 = simulate_mmap_load(&path1, &tensors1).expect("m1");
            let mut loader2 = simulate_mmap_load(&path2, &tensors2).expect("m2");

            let d1 = lazy_load_tensor(&mut loader1, &tensors1[0]).expect("l1");
            let d2 = lazy_load_tensor(&mut loader2, &tensors2[0]).expect("l2");

            prop_assert_eq!(d1.len(), d2.len());
            for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < f32::EPSILON,
                    "determinism broken at index {}: {} vs {}", i, a, b
                );
            }
        }
    }
}
