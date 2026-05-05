//! # Recipe: NF4 Lint — Codebook Integrity Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr nf4-lint --observation-file observation.json` (codebook fail)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates codebook drift detection. The QLoRA NF4 codebook is a fixed
//! constant — it is the precomputed inverse-CDF of a unit normal at 16
//! quantile points. A producer that emits a custom codebook (even one that
//! "looks reasonable") breaks bitsandbytes' inference path, because dequant
//! kernels use the canonical table by reference. The lint compares against
//! the reference table with a tight tolerance.
//!
//! ## Run Command
//! ```bash
//! cargo run --example nf4_lint_codebook_violation
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314, §2.1
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

const NF4_REF: [f64; 16] = [
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844976, -0.1848792, -0.0907983, 0.0, 0.07958029,
    0.1609302, 0.2461123, 0.3379152, 0.4407410, 0.5626170, 0.7229568, 1.0,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodebookFinding {
    pub index: usize,
    pub expected: String, // f64 → string (Eq friendliness)
    pub actual: String,
    pub abs_drift: String,
}

pub fn diff_codebook(observed: &[f64]) -> Vec<CodebookFinding> {
    if observed.len() != 16 {
        return vec![CodebookFinding {
            index: usize::MAX,
            expected: "len=16".into(),
            actual: format!("len={}", observed.len()),
            abs_drift: "n/a".into(),
        }];
    }
    let mut out = Vec::new();
    for (i, (o, r)) in observed.iter().zip(NF4_REF.iter()).enumerate() {
        let drift = (o - r).abs();
        if drift > 1e-6 {
            out.push(CodebookFinding {
                index: i,
                expected: format!("{r:.7}"),
                actual: format!("{o:.7}"),
                abs_drift: format!("{drift:.2e}"),
            });
        }
    }
    out
}

fn build_drifted_observation() -> Value {
    let mut cb = NF4_REF.to_vec();
    cb[3] = -0.40; // index 3 drifted from -0.3949175
    cb[12] = 0.45; // index 12 drifted from 0.4407410
    json!({
        "schema_version": 1,
        "codebook": cb,
        "frobenius_rel_err": 0.041,
        "storage": "packed_2x4bit_le"
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("nf4_lint_codebook_violation")?;
    let obs = build_drifted_observation();
    let cb: Vec<f64> = obs["codebook"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect();

    let findings = diff_codebook(&cb);
    println!("=== Recipe: {} ===", ctx.name());
    println!("Codebook drift: {} entries", findings.len());
    for f in &findings {
        println!(
            "  index {}: expected {}, actual {}, |drift|={}",
            f.index, f.expected, f.actual, f.abs_drift
        );
    }
    ctx.record_metric("drifted_entries", findings.len() as i64);
    ctx.record_string_metric("verdict", if findings.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_violation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn reference_codebook_has_zero_drift() {
        let f = diff_codebook(&NF4_REF);
        assert!(f.is_empty());
    }

    #[test]
    fn drifted_codebook_flags_each_drifted_index() {
        let mut cb = NF4_REF.to_vec();
        cb[3] = -0.40;
        cb[12] = 0.45;
        let f = diff_codebook(&cb);
        let indices: Vec<usize> = f.iter().map(|x| x.index).collect();
        assert_eq!(indices, vec![3, 12]);
    }

    #[test]
    fn wrong_length_yields_single_length_finding() {
        let f = diff_codebook(&NF4_REF[..15]);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].index, usize::MAX);
    }

    #[test]
    fn tolerance_below_1e_minus_6_passes() {
        // Round-trip drift inside the tolerance band must not be flagged —
        // quantile constants are stored as f64 but published to 7 decimals.
        let mut cb = NF4_REF.to_vec();
        cb[5] += 1e-8;
        let f = diff_codebook(&cb);
        assert!(f.is_empty());
    }
}
