//! # Recipe: Progressive Model Loading
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
//! Load model progressively with UI feedback.
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_progressive_loading
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_progressive_loading")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Progressive model loading simulation");
    println!();

    // Define model chunks
    let chunks = vec![
        ModelChunk {
            id: 0,
            name: "metadata".to_string(),
            size_kb: 5,
            required: true,
        },
        ModelChunk {
            id: 1,
            name: "embeddings".to_string(),
            size_kb: 200,
            required: true,
        },
        ModelChunk {
            id: 2,
            name: "layer_0".to_string(),
            size_kb: 150,
            required: true,
        },
        ModelChunk {
            id: 3,
            name: "layer_1".to_string(),
            size_kb: 150,
            required: true,
        },
        ModelChunk {
            id: 4,
            name: "layer_2".to_string(),
            size_kb: 150,
            required: true,
        },
        ModelChunk {
            id: 5,
            name: "output".to_string(),
            size_kb: 50,
            required: true,
        },
        ModelChunk {
            id: 6,
            name: "cache".to_string(),
            size_kb: 100,
            required: false,
        },
    ];

    let total_size: u32 = chunks.iter().map(|c| c.size_kb).sum();
    ctx.record_metric("total_chunks", chunks.len() as i64);
    ctx.record_metric("total_size_kb", i64::from(total_size));

    println!("Model chunks:");
    for chunk in &chunks {
        let required = if chunk.required {
            "[required]"
        } else {
            "[optional]"
        };
        println!("  {} ({}KB) {}", chunk.name, chunk.size_kb, required);
    }
    println!("  Total: {}KB", total_size);
    println!();

    // Progressive loading simulation
    let mut loader = ProgressiveLoader::new(chunks.clone());

    println!("Loading progress:");
    println!("{:-<50}", "");

    while !loader.is_complete() {
        let progress = loader.load_next()?;
        let bar = create_progress_bar(progress.percent, 30);
        println!(
            "  {} {:>3}% [{}] {}",
            progress.chunk_name, progress.percent, bar, progress.status
        );
    }
    println!("{:-<50}", "");

    // Loading statistics
    let stats = loader.get_stats();
    ctx.record_metric("load_time_ms", i64::from(stats.total_time_ms));
    ctx.record_float_metric("throughput_kbps", stats.throughput_kbps);

    println!();
    println!("Loading complete:");
    println!("  Total time: {}ms", stats.total_time_ms);
    println!("  Throughput: {:.1}KB/s", stats.throughput_kbps);
    println!(
        "  Chunks loaded: {}/{}",
        stats.chunks_loaded, stats.chunks_total
    );

    // Demonstrate early inference capability
    println!();
    println!("Early inference capability:");
    let min_required = loader.get_minimum_usable_chunks();
    println!("  Minimum chunks for inference: {}", min_required);
    println!(
        "  Can run basic inference after {}KB loaded",
        chunks
            .iter()
            .take(min_required)
            .map(|c| c.size_kb)
            .sum::<u32>()
    );

    // Save loading log
    let log_path = ctx.path("loading_log.json");
    loader.save_log(&log_path)?;
    println!();
    println!("Loading log saved to: {:?}", log_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelChunk {
    id: u32,
    name: String,
    size_kb: u32,
    required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoadProgress {
    chunk_name: String,
    percent: u32,
    bytes_loaded: u32,
    bytes_total: u32,
    status: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoadStats {
    total_time_ms: u32,
    throughput_kbps: f64,
    chunks_loaded: usize,
    chunks_total: usize,
}

#[derive(Debug)]
struct ProgressiveLoader {
    chunks: Vec<ModelChunk>,
    loaded: Vec<bool>,
    bytes_loaded: u32,
    bytes_total: u32,
    current_idx: usize,
    log: Vec<LoadProgress>,
}

impl ProgressiveLoader {
    fn new(chunks: Vec<ModelChunk>) -> Self {
        let bytes_total: u32 = chunks.iter().map(|c| c.size_kb * 1024).sum();
        let loaded = vec![false; chunks.len()];

        Self {
            chunks,
            loaded,
            bytes_loaded: 0,
            bytes_total,
            current_idx: 0,
            log: Vec::new(),
        }
    }

    fn load_next(&mut self) -> Result<LoadProgress> {
        if self.current_idx >= self.chunks.len() {
            return Err(CookbookError::invalid_format(
                "All chunks already loaded".to_string(),
            ));
        }

        let chunk = &self.chunks[self.current_idx];
        self.bytes_loaded += chunk.size_kb * 1024;
        self.loaded[self.current_idx] = true;

        let percent = ((f64::from(self.bytes_loaded) / f64::from(self.bytes_total)) * 100.0) as u32;

        let progress = LoadProgress {
            chunk_name: chunk.name.clone(),
            percent,
            bytes_loaded: self.bytes_loaded,
            bytes_total: self.bytes_total,
            status: "loaded".to_string(),
        };

        self.log.push(progress.clone());
        self.current_idx += 1;

        Ok(progress)
    }

    fn is_complete(&self) -> bool {
        self.current_idx >= self.chunks.len()
    }

    fn get_stats(&self) -> LoadStats {
        // Deterministic simulated time: 1ms per KB
        let total_time = self.bytes_loaded / 1024;
        let throughput = if total_time > 0 {
            (f64::from(self.bytes_loaded) / 1024.0) / (f64::from(total_time) / 1000.0)
        } else {
            0.0
        };

        LoadStats {
            total_time_ms: total_time,
            throughput_kbps: throughput,
            chunks_loaded: self.loaded.iter().filter(|&&l| l).count(),
            chunks_total: self.chunks.len(),
        }
    }

    fn get_minimum_usable_chunks(&self) -> usize {
        self.chunks.iter().take_while(|c| c.required).count() + 1
    }

    fn save_log(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.log)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

fn create_progress_bar(percent: u32, width: usize) -> String {
    let filled = (percent as usize * width) / 100;
    let empty = width - filled;
    format!("{}{}", "=".repeat(filled), " ".repeat(empty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let chunks = vec![ModelChunk {
            id: 0,
            name: "test".to_string(),
            size_kb: 100,
            required: true,
        }];

        let loader = ProgressiveLoader::new(chunks);

        assert!(!loader.is_complete());
        assert_eq!(loader.bytes_loaded, 0);
    }

    #[test]
    fn test_load_next() {
        let chunks = vec![ModelChunk {
            id: 0,
            name: "chunk1".to_string(),
            size_kb: 100,
            required: true,
        }];

        let mut loader = ProgressiveLoader::new(chunks);
        let progress = loader.load_next().unwrap();

        assert_eq!(progress.chunk_name, "chunk1");
        assert_eq!(progress.percent, 100);
        assert!(loader.is_complete());
    }

    #[test]
    fn test_progressive_percent() {
        let chunks = vec![
            ModelChunk {
                id: 0,
                name: "c1".to_string(),
                size_kb: 50,
                required: true,
            },
            ModelChunk {
                id: 1,
                name: "c2".to_string(),
                size_kb: 50,
                required: true,
            },
        ];

        let mut loader = ProgressiveLoader::new(chunks);

        let p1 = loader.load_next().unwrap();
        assert_eq!(p1.percent, 50);

        let p2 = loader.load_next().unwrap();
        assert_eq!(p2.percent, 100);
    }

    #[test]
    fn test_load_complete_error() {
        let chunks = vec![ModelChunk {
            id: 0,
            name: "test".to_string(),
            size_kb: 100,
            required: true,
        }];

        let mut loader = ProgressiveLoader::new(chunks);
        loader.load_next().unwrap();

        let result = loader.load_next();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_stats() {
        let chunks = vec![ModelChunk {
            id: 0,
            name: "test".to_string(),
            size_kb: 100,
            required: true,
        }];

        let mut loader = ProgressiveLoader::new(chunks);
        loader.load_next().unwrap();

        let stats = loader.get_stats();
        assert_eq!(stats.chunks_loaded, 1);
        assert_eq!(stats.chunks_total, 1);
    }

    #[test]
    fn test_progress_bar() {
        assert_eq!(create_progress_bar(50, 10), "=====     ");
        assert_eq!(create_progress_bar(100, 10), "==========");
        assert_eq!(create_progress_bar(0, 10), "          ");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_final_percent_is_100(sizes in proptest::collection::vec(1u32..100, 1..10)) {
            let chunks: Vec<_> = sizes.iter().enumerate().map(|(i, &size)| {
                ModelChunk {
                    id: i as u32,
                    name: format!("chunk{}", i),
                    size_kb: size,
                    required: true,
                }
            }).collect();

            let mut loader = ProgressiveLoader::new(chunks);
            let mut last_progress = None;

            while !loader.is_complete() {
                last_progress = Some(loader.load_next().unwrap());
            }

            if let Some(progress) = last_progress {
                prop_assert_eq!(progress.percent, 100);
            }
        }

        #[test]
        fn prop_percent_monotonic(sizes in proptest::collection::vec(1u32..50, 2..5)) {
            let chunks: Vec<_> = sizes.iter().enumerate().map(|(i, &size)| {
                ModelChunk {
                    id: i as u32,
                    name: format!("chunk{}", i),
                    size_kb: size,
                    required: true,
                }
            }).collect();

            let mut loader = ProgressiveLoader::new(chunks);
            let mut last_percent = 0u32;

            while !loader.is_complete() {
                let progress = loader.load_next().unwrap();
                prop_assert!(progress.percent >= last_percent);
                last_percent = progress.percent;
            }
        }
    }
}
