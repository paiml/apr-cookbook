//! # Recipe: Cross-Format Parity (APR vs GGUF vs SafeTensors)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr parity --ref model.apr --cmp model.gguf,model.safetensors`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example parity_format` exits 0
//! 2. [x] `cargo test --example parity_format` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr parity` in-process (no shell-out)
//! 10. [x] Unit tests cover shape match, dtype match, hash match, missing tensors
//!
//! ## Learning Objective
//! Demonstrates cross-format parity: same logical model emitted into APR,
//! GGUF, and SafeTensors containers should expose identical tensor catalogs
//! (shape + dtype + bytewise hash). The recipe synthesizes three in-memory
//! manifests and walks the intersection to compute a per-tensor parity grid.
//!
//! ## Run Command
//! ```bash
//! cargo run --example parity_format
//! ```
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. arXiv:1910.03771

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub hash_hex: String,
}

#[derive(Debug, Clone)]
pub struct FormatManifest {
    pub format: String,
    pub tensors: BTreeMap<String, TensorRec>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParityVerdict {
    Match,
    ShapeMismatch,
    DtypeMismatch,
    HashMismatch,
    MissingInCompare,
}

impl ParityVerdict {
    fn label(&self) -> &'static str {
        match self {
            ParityVerdict::Match => "match",
            ParityVerdict::ShapeMismatch => "shape-mismatch",
            ParityVerdict::DtypeMismatch => "dtype-mismatch",
            ParityVerdict::HashMismatch => "hash-mismatch",
            ParityVerdict::MissingInCompare => "missing",
        }
    }
}

pub fn compare_tensor(reference: &TensorRec, candidate: Option<&TensorRec>) -> ParityVerdict {
    let Some(c) = candidate else {
        return ParityVerdict::MissingInCompare;
    };
    if c.shape != reference.shape {
        return ParityVerdict::ShapeMismatch;
    }
    if c.dtype != reference.dtype {
        return ParityVerdict::DtypeMismatch;
    }
    if c.hash_hex != reference.hash_hex {
        return ParityVerdict::HashMismatch;
    }
    ParityVerdict::Match
}

pub fn compare_manifests(
    reference: &FormatManifest,
    candidate: &FormatManifest,
) -> BTreeMap<String, ParityVerdict> {
    let mut out = BTreeMap::new();
    let names: BTreeSet<_> = reference
        .tensors
        .keys()
        .chain(candidate.tensors.keys())
        .cloned()
        .collect();
    for n in names {
        match (reference.tensors.get(&n), candidate.tensors.get(&n)) {
            (Some(r), c) => out.insert(n, compare_tensor(r, c)),
            (None, Some(_)) => out.insert(n, ParityVerdict::MissingInCompare),
            (None, None) => None,
        };
    }
    out
}

fn tensor(shape: &[usize], dtype: &str, hash: &str) -> TensorRec {
    TensorRec {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        hash_hex: hash.to_string(),
    }
}

fn base_manifest(fmt: &str) -> FormatManifest {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "embed_tokens.weight".into(),
        tensor(&[32000, 768], "f16", "aa11"),
    );
    tensors.insert(
        "layers.0.self_attn.q_proj.weight".into(),
        tensor(&[768, 768], "f16", "bb22"),
    );
    tensors.insert(
        "layers.0.mlp.gate_proj.weight".into(),
        tensor(&[768, 2048], "f16", "cc33"),
    );
    tensors.insert("norm.weight".into(), tensor(&[768], "f16", "dd44"));
    FormatManifest {
        format: fmt.to_string(),
        tensors,
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("parity_format")?;
    println!("=== Recipe: {} ===", ctx.name());

    let apr = base_manifest("apr");
    let gguf = base_manifest("gguf");
    // SafeTensors sometimes advertises bf16 rather than f16 — triggers dtype-mismatch.
    let mut safetensors = base_manifest("safetensors");
    if let Some(t) = safetensors.tensors.get_mut("norm.weight") {
        t.dtype = "bf16".into();
    }

    let gguf_diff = compare_manifests(&apr, &gguf);
    let st_diff = compare_manifests(&apr, &safetensors);

    for (fmt, diff) in [("apr->gguf", &gguf_diff), ("apr->safetensors", &st_diff)] {
        println!("--- {} ---", fmt);
        let mut matched = 0;
        for (name, v) in diff {
            println!("  {:<40} {}", name, v.label());
            if *v == ParityVerdict::Match {
                matched += 1;
            }
        }
        println!("  matched {}/{}", matched, diff.len());
    }

    let report = json!({
        "recipe": ctx.name(),
        "apr_gguf": gguf_diff.iter().map(|(k, v)| (k.clone(), v.label())).collect::<BTreeMap<_, _>>(),
        "apr_safetensors": st_diff.iter().map(|(k, v)| (k.clone(), v.label())).collect::<BTreeMap<_, _>>(),
    });
    let path = ctx.path("parity-format.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let gguf_matched = gguf_diff
        .values()
        .filter(|v| **v == ParityVerdict::Match)
        .count();
    let st_matched = st_diff
        .values()
        .filter(|v| **v == ParityVerdict::Match)
        .count();
    ctx.record_metric("gguf_matched", gguf_matched as i64);
    ctx.record_metric("safetensors_matched", st_matched as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_tensor_matches() {
        let r = tensor(&[2, 3], "f16", "aa");
        assert_eq!(compare_tensor(&r, Some(&r.clone())), ParityVerdict::Match);
    }

    #[test]
    fn shape_mismatch_detected() {
        let r = tensor(&[2, 3], "f16", "aa");
        let c = tensor(&[2, 4], "f16", "aa");
        assert_eq!(compare_tensor(&r, Some(&c)), ParityVerdict::ShapeMismatch);
    }

    #[test]
    fn dtype_mismatch_detected() {
        let r = tensor(&[2, 3], "f16", "aa");
        let c = tensor(&[2, 3], "bf16", "aa");
        assert_eq!(compare_tensor(&r, Some(&c)), ParityVerdict::DtypeMismatch);
    }

    #[test]
    fn hash_mismatch_detected() {
        let r = tensor(&[2, 3], "f16", "aa");
        let c = tensor(&[2, 3], "f16", "bb");
        assert_eq!(compare_tensor(&r, Some(&c)), ParityVerdict::HashMismatch);
    }

    #[test]
    fn missing_detected() {
        let r = tensor(&[2, 3], "f16", "aa");
        assert_eq!(compare_tensor(&r, None), ParityVerdict::MissingInCompare);
    }
}
