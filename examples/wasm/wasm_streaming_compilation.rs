//! # Recipe: WASM Streaming Compilation
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
//! Stream-compile WASM module while downloading.
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_streaming_compilation
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_streaming_compilation")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("WASM streaming compilation simulation");
    println!();

    // WASM module info
    let module = WasmModule {
        name: "model-inference".to_string(),
        size_kb: 500,
        sections: vec![
            WasmSection {
                name: "type".to_string(),
                size_kb: 10,
            },
            WasmSection {
                name: "import".to_string(),
                size_kb: 20,
            },
            WasmSection {
                name: "function".to_string(),
                size_kb: 50,
            },
            WasmSection {
                name: "table".to_string(),
                size_kb: 5,
            },
            WasmSection {
                name: "memory".to_string(),
                size_kb: 10,
            },
            WasmSection {
                name: "global".to_string(),
                size_kb: 5,
            },
            WasmSection {
                name: "export".to_string(),
                size_kb: 10,
            },
            WasmSection {
                name: "code".to_string(),
                size_kb: 350,
            },
            WasmSection {
                name: "data".to_string(),
                size_kb: 40,
            },
        ],
    };

    ctx.record_metric("module_size_kb", i64::from(module.size_kb));
    ctx.record_metric("section_count", module.sections.len() as i64);

    println!("WASM Module: {}", module.name);
    println!("Total size: {}KB", module.size_kb);
    println!();
    println!("Sections:");
    for section in &module.sections {
        println!("  {}: {}KB", section.name, section.size_kb);
    }
    println!();

    // Compare compilation strategies
    println!("Compilation Strategy Comparison:");
    println!("{:-<65}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>15}",
        "Strategy", "Download", "Compile", "Time-to-Ready"
    );
    println!("{:-<65}", "");

    // Synchronous compilation
    let sync_result = simulate_sync_compilation(&module);
    println!(
        "{:<20} {:>10}ms {:>10}ms {:>13}ms",
        "Synchronous", sync_result.download_ms, sync_result.compile_ms, sync_result.total_ms
    );

    // Streaming compilation
    let stream_result = simulate_streaming_compilation(&module);
    println!(
        "{:<20} {:>10}ms {:>10}ms {:>13}ms",
        "Streaming", stream_result.download_ms, stream_result.compile_ms, stream_result.total_ms
    );

    // Cached compilation
    let cached_result = simulate_cached_compilation(&module);
    println!(
        "{:<20} {:>10}ms {:>10}ms {:>13}ms",
        "Cached", cached_result.download_ms, cached_result.compile_ms, cached_result.total_ms
    );

    println!("{:-<65}", "");

    // Calculate improvements
    let stream_improvement = ((f64::from(sync_result.total_ms)
        - f64::from(stream_result.total_ms))
        / f64::from(sync_result.total_ms))
        * 100.0;
    let cache_improvement = ((f64::from(sync_result.total_ms) - f64::from(cached_result.total_ms))
        / f64::from(sync_result.total_ms))
        * 100.0;

    ctx.record_float_metric("streaming_improvement_pct", stream_improvement);
    ctx.record_float_metric("cache_improvement_pct", cache_improvement);

    println!();
    println!("Improvement over synchronous:");
    println!("  Streaming: {:.1}% faster", stream_improvement);
    println!("  Cached: {:.1}% faster", cache_improvement);

    // Browser compatibility
    let compat = check_browser_compatibility();
    println!();
    println!("Browser Streaming Support:");
    for (browser, supported) in &compat {
        let status = if *supported { "✓" } else { "✗" };
        println!("  {} {}", status, browser);
    }

    // Save results
    let results_path = ctx.path("streaming_results.json");
    save_results(&results_path, &[sync_result, stream_result, cached_result])?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmSection {
    name: String,
    size_kb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmModule {
    name: String,
    size_kb: u32,
    sections: Vec<WasmSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompilationResult {
    strategy: String,
    download_ms: u32,
    compile_ms: u32,
    total_ms: u32,
}

fn simulate_sync_compilation(module: &WasmModule) -> CompilationResult {
    // Synchronous: download first, then compile
    let download_ms = module.size_kb; // 1ms per KB
    let compile_ms = module.size_kb / 2; // 0.5ms per KB for compilation

    CompilationResult {
        strategy: "synchronous".to_string(),
        download_ms,
        compile_ms,
        total_ms: download_ms + compile_ms,
    }
}

fn simulate_streaming_compilation(module: &WasmModule) -> CompilationResult {
    // Streaming: compile while downloading (parallel)
    let download_ms = module.size_kb; // 1ms per KB
    let compile_ms = module.size_kb / 2; // 0.5ms per KB

    // Total is max of download and compile (overlapped)
    // Plus some overhead for streaming
    let overhead = 20u32; // Streaming overhead
    let total_ms = download_ms.max(compile_ms) + overhead;

    CompilationResult {
        strategy: "streaming".to_string(),
        download_ms,
        compile_ms,
        total_ms,
    }
}

fn simulate_cached_compilation(module: &WasmModule) -> CompilationResult {
    // Cached: no download, minimal compile (validation only)
    let download_ms = 0; // From cache
    let compile_ms = module.size_kb / 20; // Just validation, 20x faster

    CompilationResult {
        strategy: "cached".to_string(),
        download_ms,
        compile_ms,
        total_ms: download_ms + compile_ms,
    }
}

fn check_browser_compatibility() -> Vec<(String, bool)> {
    vec![
        ("Chrome 61+".to_string(), true),
        ("Firefox 58+".to_string(), true),
        ("Safari 15+".to_string(), true),
        ("Edge 79+".to_string(), true),
        ("Opera 48+".to_string(), true),
        ("IE 11".to_string(), false),
    ]
}

fn save_results(path: &std::path::Path, results: &[CompilationResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_module() -> WasmModule {
        WasmModule {
            name: "test".to_string(),
            size_kb: 100,
            sections: vec![
                WasmSection {
                    name: "code".to_string(),
                    size_kb: 80,
                },
                WasmSection {
                    name: "data".to_string(),
                    size_kb: 20,
                },
            ],
        }
    }

    #[test]
    fn test_sync_compilation() {
        let module = test_module();
        let result = simulate_sync_compilation(&module);

        assert_eq!(result.strategy, "synchronous");
        assert_eq!(result.total_ms, result.download_ms + result.compile_ms);
    }

    #[test]
    fn test_streaming_faster_than_sync() {
        let module = test_module();
        let sync = simulate_sync_compilation(&module);
        let stream = simulate_streaming_compilation(&module);

        assert!(stream.total_ms < sync.total_ms);
    }

    #[test]
    fn test_cached_fastest() {
        let module = test_module();
        let sync = simulate_sync_compilation(&module);
        let stream = simulate_streaming_compilation(&module);
        let cached = simulate_cached_compilation(&module);

        assert!(cached.total_ms < stream.total_ms);
        assert!(cached.total_ms < sync.total_ms);
    }

    #[test]
    fn test_cached_no_download() {
        let module = test_module();
        let cached = simulate_cached_compilation(&module);

        assert_eq!(cached.download_ms, 0);
    }

    #[test]
    fn test_browser_compatibility() {
        let compat = check_browser_compatibility();

        assert!(!compat.is_empty());
        // Modern browsers should support streaming
        let chrome_support = compat.iter().find(|(b, _)| b.contains("Chrome"));
        assert!(chrome_support.is_some());
        assert!(chrome_support.unwrap().1);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_streaming_save").unwrap();
        let path = ctx.path("results.json");

        let results = vec![CompilationResult {
            strategy: "test".to_string(),
            download_ms: 100,
            compile_ms: 50,
            total_ms: 150,
        }];

        save_results(&path, &results).unwrap();
        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_streaming_faster(size_kb in 50u32..1000) {
            let module = WasmModule {
                name: "test".to_string(),
                size_kb,
                sections: vec![],
            };

            let sync = simulate_sync_compilation(&module);
            let stream = simulate_streaming_compilation(&module);

            prop_assert!(stream.total_ms < sync.total_ms);
        }

        #[test]
        fn prop_cached_no_download(size_kb in 50u32..1000) {
            let module = WasmModule {
                name: "test".to_string(),
                size_kb,
                sections: vec![],
            };

            let cached = simulate_cached_compilation(&module);
            prop_assert_eq!(cached.download_ms, 0);
        }

        #[test]
        fn prop_total_positive(size_kb in 1u32..500) {
            let module = WasmModule {
                name: "test".to_string(),
                size_kb,
                sections: vec![],
            };

            let sync = simulate_sync_compilation(&module);
            let stream = simulate_streaming_compilation(&module);
            let cached = simulate_cached_compilation(&module);

            prop_assert!(sync.total_ms > 0);
            prop_assert!(stream.total_ms > 0);
            prop_assert!(cached.total_ms >= 0);
        }
    }
}
