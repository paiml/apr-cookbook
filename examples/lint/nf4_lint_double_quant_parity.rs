//! # Recipe: NF4 Lint — Double-Quantization Parity
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr nf4-lint --observation-file observation.json` (double-quant path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the double-quantization (DQ) parity rule. NF4 stores per-block
//! scale factors; when `double_quant=true`, those scales themselves are
//! quantized (FP32 → FP8) for an additional ~0.4 bits/param savings. The
//! lint enforces three parity rules:
//!  - if `double_quant=true`, `dq_block_size` MUST be present and a power of 2
//!  - if `double_quant=true`, the inner-scale FP32 absmax MUST be recorded
//!  - if `double_quant=false`, `dq_block_size` MUST be absent (implicit zero)
//!
//! ## Run Command
//! ```bash
//! cargo run --example nf4_lint_double_quant_parity
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314, §3.2 ("Double Quantization")
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: &'static str,
    pub message: String,
}

pub fn lint_double_quant(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();
    let dq = obs
        .get("double_quant")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let dq_block = obs.get("dq_block_size").and_then(Value::as_u64);
    let dq_absmax = obs.get("dq_inner_absmax").and_then(Value::as_f64);

    if dq {
        match dq_block {
            Some(b) if b > 0 && b.is_power_of_two() => {}
            Some(b) => out.push(LintFinding {
                rule: "NF4-DQ-001",
                message: format!("dq_block_size={b} must be a power of 2"),
            }),
            None => out.push(LintFinding {
                rule: "NF4-DQ-001",
                message: "dq_block_size required when double_quant=true".into(),
            }),
        }
        match dq_absmax {
            Some(a) if a.is_finite() && a > 0.0 => {}
            _ => out.push(LintFinding {
                rule: "NF4-DQ-002",
                message: "dq_inner_absmax required (finite, > 0) when double_quant=true".into(),
            }),
        }
    } else if dq_block.is_some() || dq_absmax.is_some() {
        out.push(LintFinding {
            rule: "NF4-DQ-003",
            message: "dq_block_size / dq_inner_absmax must be absent when double_quant=false"
                .into(),
        });
    }

    out
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("nf4_lint_double_quant_parity")?;

    let happy_dq = json!({
        "double_quant": true,
        "dq_block_size": 256,
        "dq_inner_absmax": 0.91
    });
    let happy_no_dq = json!({ "double_quant": false });

    println!("=== Recipe: {} ===", ctx.name());
    println!("happy DQ:    {:?}", lint_double_quant(&happy_dq));
    println!("happy no-DQ: {:?}", lint_double_quant(&happy_no_dq));

    ctx.record_string_metric("verdict", "PASS");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_quant_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn dq_with_power_of_two_block_passes() {
        let obs = json!({
            "double_quant": true,
            "dq_block_size": 256,
            "dq_inner_absmax": 0.91
        });
        assert!(lint_double_quant(&obs).is_empty());
    }

    #[test]
    fn dq_without_block_size_fails() {
        let obs = json!({ "double_quant": true, "dq_inner_absmax": 0.91 });
        let f = lint_double_quant(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-DQ-001"));
    }

    #[test]
    fn dq_with_non_power_of_two_block_fails() {
        let obs = json!({
            "double_quant": true,
            "dq_block_size": 100,
            "dq_inner_absmax": 0.91
        });
        let f = lint_double_quant(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-DQ-001"));
    }

    #[test]
    fn no_dq_with_block_size_set_fails() {
        // Catch the "I forgot to clear the field after disabling DQ" footgun.
        let obs = json!({ "double_quant": false, "dq_block_size": 256 });
        let f = lint_double_quant(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-DQ-003"));
    }

    #[test]
    fn no_dq_with_clean_state_passes() {
        let obs = json!({ "double_quant": false });
        assert!(lint_double_quant(&obs).is_empty());
    }

    #[test]
    fn dq_with_zero_absmax_fails() {
        let obs = json!({
            "double_quant": true,
            "dq_block_size": 256,
            "dq_inner_absmax": 0.0
        });
        let f = lint_double_quant(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-DQ-002"));
    }
}
