//! # Recipe: Size-Optimized Compile
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr compile model.apr --opt-level Oz --strip --dead-kernel-elim`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example compile_size_optimized` exits 0
//! 2. [x] `cargo test --example compile_size_optimized` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr compile` size optimizations in-process (no shell-out)
//! 10. [x] Unit tests cover strip, dead-kernel-elim, compression math
//!
//! ## Learning Objective
//! Compares compiled artifact size across optimization levels (O0 - Oz). Models
//! three size-shrinking passes: symbol stripping, dead-kernel elimination, and
//! Zstd compression of the emitted source. This demonstrates the size/quality
//! tradeoff curve that `apr compile` navigates.
//!
//! ## Run Command
//! ```bash
//! cargo run --example compile_size_optimized
//! ```
//!
//! ## References
//! - Lattner, C. et al. (2021). *MLIR: A Compiler Infrastructure for the End of Moore's Law*. CGO. arXiv:2002.11054

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptLevel {
    O0,
    O1,
    O2,
    O3,
    Oz,
}

impl OptLevel {
    fn label(self) -> &'static str {
        match self {
            Self::O0 => "O0",
            Self::O1 => "O1",
            Self::O2 => "O2",
            Self::O3 => "O3",
            Self::Oz => "Oz",
        }
    }
    fn strip_symbols(self) -> bool {
        matches!(self, Self::O3 | Self::Oz)
    }
    fn dead_kernel_elim(self) -> bool {
        matches!(self, Self::O2 | Self::O3 | Self::Oz)
    }
    fn compress(self) -> bool {
        matches!(self, Self::Oz)
    }
}

#[derive(Debug, Clone)]
struct Kernel {
    name: String,
    body: Vec<u8>,
    reachable: bool,
}

#[derive(Debug, Clone)]
struct CompileResult {
    opt_level: OptLevel,
    kept_kernels: usize,
    bytes_before_compress: usize,
    bytes_after_compress: usize,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn build_kernel_set(seed: u64, n: usize) -> Vec<Kernel> {
    let bytes = generate_model_payload(seed, n * 256);
    (0..n)
        .map(|i| {
            let body = bytes[i * 256..(i + 1) * 256].to_vec();
            // Mark every 3rd kernel as unreachable (dead code).
            let reachable = (i % 3) != 2;
            Kernel {
                name: format!("apr_kernel_{i}"),
                body,
                reachable,
            }
        })
        .collect()
}

fn emit_pseudo_source(kernels: &[Kernel], include_symbols: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"// APR compiled module\n");
    for k in kernels {
        if include_symbols {
            out.extend_from_slice(format!("// symbol: {}\n", k.name).as_bytes());
        }
        out.extend_from_slice(format!(".entry {} {{\n", k.name).as_bytes());
        out.extend_from_slice(&k.body);
        out.extend_from_slice(b"\n}\n");
    }
    out
}

fn compile_with(level: OptLevel, kernels: &[Kernel]) -> Result<CompileResult> {
    // Dead-kernel elimination.
    let filtered: Vec<&Kernel> = if level.dead_kernel_elim() {
        kernels.iter().filter(|k| k.reachable).collect()
    } else {
        kernels.iter().collect()
    };
    let filtered_owned: Vec<Kernel> = filtered.iter().map(|k| (*k).clone()).collect();

    // Symbol stripping.
    let src = emit_pseudo_source(&filtered_owned, !level.strip_symbols());
    let bytes_before = src.len();

    // Optional Zstd compression.
    let bytes_after = if level.compress() {
        zstd::stream::encode_all(src.as_slice(), 3)
            .map_err(|e| CookbookError::invalid_format(format!("zstd: {e}")))?
            .len()
    } else {
        bytes_before
    };

    Ok(CompileResult {
        opt_level: level,
        kept_kernels: filtered_owned.len(),
        bytes_before_compress: bytes_before,
        bytes_after_compress: bytes_after,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("compile_size_optimized")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seed = hash_name_to_seed("compile-size");
    let kernels = build_kernel_set(seed, 12);
    let reachable = kernels.iter().filter(|k| k.reachable).count();
    println!(
        "Input: {} kernels, {} reachable, {} dead",
        kernels.len(),
        reachable,
        kernels.len() - reachable
    );

    let levels = [
        OptLevel::O0,
        OptLevel::O1,
        OptLevel::O2,
        OptLevel::O3,
        OptLevel::Oz,
    ];
    let mut results = Vec::new();
    for &l in &levels {
        results.push(compile_with(l, &kernels)?);
    }

    println!("\n--- Compile Result by Opt Level ---");
    println!(
        "{:>4} {:>10} {:>18} {:>18} {:>10}",
        "Lvl", "Kernels", "Before", "After", "Ratio"
    );
    let baseline_after = results[0].bytes_after_compress.max(1) as f64;
    for r in &results {
        let ratio = r.bytes_after_compress as f64 / baseline_after;
        println!(
            "{:>4} {:>10} {:>18} {:>18} {:>10.2}x",
            r.opt_level.label(),
            r.kept_kernels,
            r.bytes_before_compress,
            r.bytes_after_compress,
            ratio
        );
    }

    // Sanity: Oz must be <= O0 in bytes after compression.
    let o0 = &results[0];
    let oz = results
        .last()
        .ok_or_else(|| CookbookError::invalid_format("missing Oz"))?;
    assert!(oz.bytes_after_compress <= o0.bytes_after_compress);

    let out = json!({
        "recipe": ctx.name(),
        "input_kernels": kernels.len(),
        "reachable_kernels": reachable,
        "results": results.iter().map(|r| json!({
            "opt_level": r.opt_level.label(),
            "kept_kernels": r.kept_kernels,
            "bytes_before": r.bytes_before_compress,
            "bytes_after": r.bytes_after_compress,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("compile-sizes.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_level_properties() {
        assert!(!OptLevel::O0.strip_symbols());
        assert!(OptLevel::O3.strip_symbols());
        assert!(OptLevel::Oz.compress());
        assert!(!OptLevel::O3.compress());
    }

    #[test]
    fn test_dead_kernel_elim_drops_some() {
        let kernels = build_kernel_set(hash_name_to_seed("test-elim"), 9);
        let o0 = compile_with(OptLevel::O0, &kernels).expect("compile O0");
        let o2 = compile_with(OptLevel::O2, &kernels).expect("compile O2");
        assert!(o2.kept_kernels < o0.kept_kernels);
    }

    #[test]
    fn test_strip_reduces_size_before_compress() {
        let kernels = build_kernel_set(hash_name_to_seed("test-strip"), 4);
        let o1 = compile_with(OptLevel::O1, &kernels).expect("compile O1");
        let o3 = compile_with(OptLevel::O3, &kernels).expect("compile O3");
        // O3 strips symbols and eliminates dead code.
        assert!(o3.bytes_before_compress <= o1.bytes_before_compress);
    }

    #[test]
    fn test_oz_compresses_to_smaller_after_bytes() {
        let kernels = build_kernel_set(hash_name_to_seed("test-oz"), 8);
        let oz = compile_with(OptLevel::Oz, &kernels).expect("compile Oz");
        assert!(oz.bytes_after_compress <= oz.bytes_before_compress);
    }

    #[test]
    fn test_deterministic_bytes() {
        let k1 = build_kernel_set(12345, 6);
        let k2 = build_kernel_set(12345, 6);
        let r1 = compile_with(OptLevel::Oz, &k1).expect("c1");
        let r2 = compile_with(OptLevel::Oz, &k2).expect("c2");
        assert_eq!(r1.bytes_after_compress, r2.bytes_after_compress);
    }
}
