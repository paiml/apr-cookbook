//! # Recipe: Compile APR Model to PTX Target
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr compile model.apr --target ptx --sm-arch sm_80`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example compile_ptx` exits 0
//! 2. [x] `cargo test --example compile_ptx` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr compile` PTX pipeline in-process (no shell-out)
//! 10. [x] Unit tests cover arch detection, directive emission, size accounting
//!
//! ## Learning Objective
//! Demonstrates AOT compilation targeting NVIDIA PTX. Produces a minimal, valid
//! PTX header (.version / .target / .address_size) plus a kernel stub, then
//! records artifact size and the detected SM architecture.
//!
//! ## Run Command
//! ```bash
//! cargo run --example compile_ptx
//! ```
//!
//! ## Format Variants
//! ```bash
//! apr compile model.apr --target ptx --sm-arch sm_80
//! apr compile model.gguf --target ptx --sm-arch sm_90
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
enum SmArch {
    Sm70,
    Sm75,
    Sm80,
    Sm86,
    Sm89,
    Sm90,
}

impl SmArch {
    fn directive(self) -> &'static str {
        match self {
            Self::Sm70 => "sm_70",
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
        }
    }
    fn ptx_version(self) -> &'static str {
        match self {
            Self::Sm70 => "6.0",
            Self::Sm75 => "6.4",
            Self::Sm80 => "7.0",
            Self::Sm86 => "7.1",
            Self::Sm89 => "7.8",
            Self::Sm90 => "8.0",
        }
    }
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "sm_70" => Self::Sm70,
            "sm_75" => Self::Sm75,
            "sm_80" => Self::Sm80,
            "sm_86" => Self::Sm86,
            "sm_89" => Self::Sm89,
            "sm_90" => Self::Sm90,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
struct PtxArtifact {
    arch: SmArch,
    source: String,
    size_bytes: usize,
    n_kernels: usize,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn emit_ptx(arch: SmArch, kernel_names: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(&format!("//\n// APR -> PTX {}\n//\n", arch.directive()));
    out.push_str(&format!(".version {}\n", arch.ptx_version()));
    out.push_str(&format!(".target {}\n", arch.directive()));
    out.push_str(".address_size 64\n\n");
    for (i, name) in kernel_names.iter().enumerate() {
        out.push_str(&format!(".visible .entry {} (\n", name));
        out.push_str("    .param .u64 input,\n");
        out.push_str("    .param .u64 output,\n");
        out.push_str("    .param .u32 n\n");
        out.push_str(")\n");
        out.push_str("{\n");
        out.push_str(&format!("    // kernel {} body stub\n", i));
        out.push_str("    ret;\n");
        out.push_str("}\n\n");
    }
    out
}

fn compile_to_ptx(arch: SmArch, kernel_names: &[&str]) -> PtxArtifact {
    let source = emit_ptx(arch, kernel_names);
    PtxArtifact {
        arch,
        size_bytes: source.len(),
        source,
        n_kernels: kernel_names.len(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("compile_ptx")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Build a small representative model as the "source".
    let dim = 16;
    let seed = hash_name_to_seed("compile-ptx");
    let model_bytes = generate_model_payload(seed, dim * dim);
    let model_path = ctx.path("input.apr");
    std::fs::write(&model_path, &model_bytes)?;

    // Compile for sm_80 (A100 class).
    let arch_str = "sm_80";
    let arch = SmArch::from_str(arch_str)
        .ok_or_else(|| CookbookError::invalid_format(format!("unknown sm arch: {arch_str}")))?;
    let kernels = ["apr_matmul", "apr_softmax", "apr_layernorm"];
    let artifact = compile_to_ptx(arch, &kernels);

    println!(
        "Target: {}, PTX version: {}",
        artifact.arch.directive(),
        arch.ptx_version()
    );
    println!("Kernels: {}", artifact.n_kernels);
    println!("Emitted bytes: {}", artifact.size_bytes);

    let ptx_path = ctx.path("apr_kernels.ptx");
    std::fs::write(&ptx_path, artifact.source.as_bytes())?;

    // Sanity.
    assert!(artifact.source.contains(".target sm_80"));
    assert!(artifact.source.contains(".address_size 64"));
    for k in &kernels {
        assert!(artifact.source.contains(&format!(".visible .entry {}", k)));
    }

    let out = json!({
        "recipe": ctx.name(),
        "arch": arch.directive(),
        "ptx_version": arch.ptx_version(),
        "n_kernels": artifact.n_kernels,
        "size_bytes": artifact.size_bytes,
        "model_size_bytes": model_bytes.len(),
    });
    let out_path = ctx.path("compile-ptx.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.record_string_metric("arch", arch.directive());
    ctx.record_metric("n_kernels", artifact.n_kernels as i64);
    ctx.record_metric("ptx_bytes", artifact.size_bytes as i64);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_from_str_roundtrip() {
        for s in ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"] {
            let a = SmArch::from_str(s).expect("parse");
            assert_eq!(a.directive(), s);
        }
    }

    #[test]
    fn test_arch_from_str_unknown() {
        assert!(SmArch::from_str("sm_999").is_none());
    }

    #[test]
    fn test_emit_ptx_header_lines() {
        let src = emit_ptx(SmArch::Sm80, &["kern"]);
        assert!(src.starts_with("//"));
        assert!(src.contains(".version 7.0"));
        assert!(src.contains(".target sm_80"));
        assert!(src.contains(".address_size 64"));
    }

    #[test]
    fn test_emit_ptx_kernel_names() {
        let src = emit_ptx(SmArch::Sm90, &["apr_a", "apr_b"]);
        assert!(src.contains(".visible .entry apr_a"));
        assert!(src.contains(".visible .entry apr_b"));
    }

    #[test]
    fn test_compile_records_size() {
        let art = compile_to_ptx(SmArch::Sm80, &["k"]);
        assert!(art.size_bytes > 50);
        assert_eq!(art.n_kernels, 1);
    }

    #[test]
    fn test_compile_empty_kernel_list() {
        let art = compile_to_ptx(SmArch::Sm80, &[]);
        assert_eq!(art.n_kernels, 0);
        // Still contains header.
        assert!(art.source.contains(".target sm_80"));
    }
}
