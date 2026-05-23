//! # Recipe: Embeddings Lint — Per-Batch Dimension Consistency
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr embeddings-lint --observation-file observation.json` (dim path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates batch-level dimension consistency. Within one
//! `/v1/embeddings` response, **every** `data[*].embedding` must have the
//! same dimension. A mixed batch indicates the producer concatenated
//! responses from two different model versions (model swap during
//! warm-up), or used Matryoshka truncation inconsistently. The lint
//! reports the histogram of observed dimensions so the operator can see
//! exactly which entries diverge.
//!
//! ## Run Command
//! ```bash
//! cargo run --example embeddings_lint_dim_consistency
//! ```
//!
//! ## References
//! - Kusupati et al. (2022). *Matryoshka Representation Learning*. arXiv:2205.13147
//! - aprender CRUX-C-13 (per-batch invariance).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimReport {
    pub histogram: BTreeMap<usize, usize>, // dim → count
    pub canonical: Option<usize>,          // most common dim
    pub passes: bool,
}

pub fn audit_dim_consistency(resp: &Value) -> DimReport {
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    if let Some(arr) = resp.get("data").and_then(Value::as_array) {
        for e in arr {
            if let Some(v) = e.get("embedding").and_then(Value::as_array) {
                *hist.entry(v.len()).or_insert(0) += 1;
            }
        }
    }
    let canonical = hist.iter().max_by_key(|(_, c)| *c).map(|(d, _)| *d);
    let passes = hist.len() <= 1;
    DimReport {
        histogram: hist,
        canonical,
        passes,
    }
}

fn build_consistent_response() -> Value {
    json!({
        "data": [
            { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] },
            { "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8] },
            { "index": 2, "embedding": [0.9, 0.0, 0.1, 0.2] }
        ]
    })
}

fn build_mixed_response() -> Value {
    json!({
        "data": [
            { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] },         // dim 4
            { "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8, 0.9] },    // dim 5 ⚠
            { "index": 2, "embedding": [0.9, 0.0, 0.1, 0.2] },         // dim 4
            { "index": 3, "embedding": [0.3] }                          // dim 1 ⚠
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("embeddings_lint_dim_consistency")?;
    println!("=== Recipe: {} ===", ctx.name());

    let consistent = audit_dim_consistency(&build_consistent_response());
    println!(
        "consistent: hist={:?}, canonical={:?}, pass={}",
        consistent.histogram, consistent.canonical, consistent.passes
    );

    let mixed = audit_dim_consistency(&build_mixed_response());
    println!(
        "mixed:      hist={:?}, canonical={:?}, pass={}",
        mixed.histogram, mixed.canonical, mixed.passes
    );

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn consistent_response_passes() {
        let r = audit_dim_consistency(&build_consistent_response());
        assert!(r.passes);
        assert_eq!(r.canonical, Some(4));
        assert_eq!(r.histogram.len(), 1);
    }

    #[test]
    fn mixed_response_fails_with_three_distinct_dims() {
        let r = audit_dim_consistency(&build_mixed_response());
        assert!(!r.passes);
        assert_eq!(r.histogram.len(), 3); // dims 1, 4, 5
        assert_eq!(r.canonical, Some(4)); // dim 4 occurs twice
    }

    #[test]
    fn empty_response_passes_vacuously() {
        // No embeddings → nothing to check, vacuous pass.
        let resp = json!({ "data": [] });
        let r = audit_dim_consistency(&resp);
        assert!(r.passes);
        assert_eq!(r.canonical, None);
    }

    #[test]
    fn missing_data_array_passes_vacuously() {
        let r = audit_dim_consistency(&json!({}));
        assert!(r.passes);
        assert_eq!(r.canonical, None);
    }

    #[test]
    fn single_embedding_always_passes() {
        let resp = json!({
            "data": [{ "index": 0, "embedding": [1.0, 2.0, 3.0] }]
        });
        let r = audit_dim_consistency(&resp);
        assert!(r.passes);
        assert_eq!(r.canonical, Some(3));
    }
}
