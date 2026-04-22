//! # Recipe: Mixed-Precision Quantization (Different Tensor Groups → Different Bits)
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr quantize --plan mixed.json model.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example quantize_mixed_precision` exits 0
//! 2. [x] `cargo test --example quantize_mixed_precision` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr quantize --plan` in-process (no shell-out)
//! 10. [x] Unit tests cover per-group plan, size reduction, role matching
//!
//! ## Learning Objective
//! Demonstrates mixed-precision quantization: different tensor roles
//! (embedding, attention, mlp, output) get assigned different bit widths via
//! a plan file. Reports per-group size reduction and aggregate model footprint.
//! Mirrors `apr quantize --plan mixed.json`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example quantize_mixed_precision
//! ```
//!
//! ## References
//! - Micikevicius, P. et al. (2018). *Mixed Precision Training*. ICLR. arXiv:1710.03740

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Embedding,
    Attention,
    Mlp,
    Output,
    Other,
}

impl Role {
    pub fn label(&self) -> &'static str {
        match self {
            Role::Embedding => "embedding",
            Role::Attention => "attention",
            Role::Mlp => "mlp",
            Role::Output => "output",
            Role::Other => "other",
        }
    }
}

pub fn classify_role(tensor_name: &str) -> Role {
    if tensor_name.contains("embed_tokens") || tensor_name.contains("wte") {
        Role::Embedding
    } else if tensor_name.contains("self_attn") || tensor_name.contains("attention") {
        Role::Attention
    } else if tensor_name.contains("mlp") || tensor_name.contains("ffn") {
        Role::Mlp
    } else if tensor_name.contains("lm_head") || tensor_name.contains("output") {
        Role::Output
    } else {
        Role::Other
    }
}

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub n_elements: u64,
}

#[derive(Debug, Clone)]
pub struct GroupReport {
    pub role: Role,
    pub bits: u32,
    pub tensors: usize,
    pub elements: u64,
    pub fp16_bytes: u64,
    pub quantized_bytes: u64,
}

pub fn plan_bits(role: Role) -> u32 {
    match role {
        Role::Embedding => 8,
        Role::Attention => 4,
        Role::Mlp => 3,
        Role::Output => 8,
        Role::Other => 4,
    }
}

pub fn run_mixed_precision(tensors: &[TensorEntry]) -> (Vec<GroupReport>, u64, u64) {
    let mut groups: BTreeMap<Role, (Vec<&TensorEntry>, u64)> = BTreeMap::new();
    for t in tensors {
        let role = classify_role(&t.name);
        let entry = groups.entry(role).or_insert((Vec::new(), 0));
        entry.0.push(t);
        entry.1 += t.n_elements;
    }
    let mut reports = Vec::new();
    let mut total_fp16: u64 = 0;
    let mut total_quant: u64 = 0;
    for (role, (ts, n_elems)) in groups {
        let bits = plan_bits(role);
        let fp16 = n_elems * 2;
        let q = (n_elems * u64::from(bits)).div_ceil(8);
        total_fp16 += fp16;
        total_quant += q;
        reports.push(GroupReport {
            role,
            bits,
            tensors: ts.len(),
            elements: n_elems,
            fp16_bytes: fp16,
            quantized_bytes: q,
        });
    }
    (reports, total_fp16, total_quant)
}

fn demo_tensors() -> Vec<TensorEntry> {
    let mut out = Vec::new();
    out.push(TensorEntry {
        name: "model.embed_tokens.weight".into(),
        n_elements: 32_000 * 768,
    });
    for i in 0..4 {
        out.push(TensorEntry {
            name: format!("model.layers.{}.self_attn.q_proj.weight", i),
            n_elements: 768 * 768,
        });
        out.push(TensorEntry {
            name: format!("model.layers.{}.mlp.gate_proj.weight", i),
            n_elements: 768 * 3072,
        });
    }
    out.push(TensorEntry {
        name: "lm_head.weight".into(),
        n_elements: 32_000 * 768,
    });
    out
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("quantize_mixed_precision")?;
    println!("=== Recipe: {} ===", ctx.name());

    let tensors = demo_tensors();
    let (reports, total_fp16, total_quant) = run_mixed_precision(&tensors);

    println!(
        "{:<10} {:>3} {:>6} {:>14} {:>14} {:>14}",
        "role", "bits", "tensors", "elements", "fp16_bytes", "quant_bytes"
    );
    for r in &reports {
        println!(
            "{:<10} {:>3} {:>6} {:>14} {:>14} {:>14}",
            r.role.label(),
            r.bits,
            r.tensors,
            r.elements,
            r.fp16_bytes,
            r.quantized_bytes
        );
    }
    let reduction = if total_fp16 > 0 {
        100.0 * (total_fp16 as f64 - total_quant as f64) / total_fp16 as f64
    } else {
        0.0
    };
    println!(
        "fp16_bytes={} quant_bytes={} reduction={:.1}%",
        total_fp16, total_quant, reduction
    );

    let report_json = json!({
        "recipe": ctx.name(),
        "total_fp16_bytes": total_fp16,
        "total_quant_bytes": total_quant,
        "reduction_pct": reduction,
        "groups": reports.iter().map(|r| json!({
            "role": r.role.label(),
            "bits": r.bits,
            "tensors": r.tensors,
            "elements": r.elements,
            "fp16_bytes": r.fp16_bytes,
            "quantized_bytes": r.quantized_bytes,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("quantize-mixed.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report_json)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("total_fp16_bytes", total_fp16 as i64);
    ctx.record_metric("total_quant_bytes", total_quant as i64);
    ctx.record_float_metric("reduction_pct", reduction);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_embedding() {
        assert_eq!(classify_role("model.embed_tokens.weight"), Role::Embedding);
    }

    #[test]
    fn classify_attention() {
        assert_eq!(
            classify_role("model.layers.0.self_attn.q_proj.weight"),
            Role::Attention
        );
    }

    #[test]
    fn classify_mlp() {
        assert_eq!(
            classify_role("model.layers.0.mlp.gate_proj.weight"),
            Role::Mlp
        );
    }

    #[test]
    fn mixed_precision_shrinks_model() {
        let (_, fp16, q) = run_mixed_precision(&demo_tensors());
        assert!(q < fp16);
    }

    #[test]
    fn plan_bits_distinct_per_role() {
        assert_ne!(plan_bits(Role::Embedding), plan_bits(Role::Mlp));
        assert!(plan_bits(Role::Mlp) < plan_bits(Role::Embedding));
    }
}
