//! # Recipe: Flow — Parameter Count Sweep over Layer Depths
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr flow model.apr --depths 1,2,4,8,16,24,32,48,64,96 --params`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example flow_depth_sweep` exits 0
//! 2. [x] `cargo test --example flow_depth_sweep` passes
//! 3. [x] Deterministic output (pure arithmetic)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr flow --depths` in-process (no shell-out)
//! 10. [x] Unit tests cover linearity, zero-depth, scaling law
//!
//! ## Learning Objective
//! Computes total parameter count over a sweep of 10 transformer depths. Each
//! layer contributes attention (4 × d²) + FFN (8 × d²) + layer-norm params,
//! and we verify params scale linearly with depth at fixed hidden size.
//!
//! ## Run Command
//! ```bash
//! cargo run --example flow_depth_sweep
//! ```
//!
//! ## References
//! - Cytron, R. et al. (1991). *Efficiently Computing SSA Form*. TOPLAS. DOI: 10.1145/115372.115320

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct DepthRow {
    n_layers: usize,
    params_total: u64,
    params_per_layer: u64,
}

/// Parameter count for one transformer block at hidden size d:
/// attention = 4 d²   (Q, K, V, O)
/// ffn       = 8 d²   (4d expansion + down)
/// layer-norm= 4 d    (two norms × gamma+beta)
fn params_per_layer(hidden: u64) -> u64 {
    4 * hidden * hidden + 8 * hidden * hidden + 4 * hidden
}

fn embedding_params(vocab: u64, hidden: u64) -> u64 {
    vocab * hidden
}

fn total_params(n_layers: usize, hidden: u64, vocab: u64) -> u64 {
    embedding_params(vocab, hidden) + n_layers as u64 * params_per_layer(hidden)
}

fn sweep(depths: &[usize], hidden: u64, vocab: u64) -> Vec<DepthRow> {
    let per = params_per_layer(hidden);
    depths
        .iter()
        .map(|&d| DepthRow {
            n_layers: d,
            params_total: total_params(d, hidden, vocab),
            params_per_layer: per,
        })
        .collect()
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("flow_depth_sweep")?;
    println!("=== Recipe: {} ===", ctx.name());

    let hidden = 768_u64;
    let vocab = 32_000_u64;
    let depths: Vec<usize> = vec![1, 2, 4, 8, 16, 24, 32, 48, 64, 96];

    let rows = sweep(&depths, hidden, vocab);
    println!(
        "\nhidden={}  vocab={}  embedding_params={}",
        hidden,
        vocab,
        embedding_params(vocab, hidden)
    );
    println!(
        "{:>6} {:>14} {:>14}",
        "Depth", "Params/layer", "Params total"
    );
    for r in &rows {
        println!(
            "{:>6} {:>14} {:>14}",
            r.n_layers, r.params_per_layer, r.params_total
        );
    }

    // Slope check between last two rows (should equal params_per_layer).
    if rows.len() >= 2 {
        let r_last = &rows[rows.len() - 1];
        let r_prev = &rows[rows.len() - 2];
        let delta_params = r_last.params_total - r_prev.params_total;
        let delta_depth = (r_last.n_layers - r_prev.n_layers) as u64;
        let slope = delta_params / delta_depth;
        println!(
            "\nSlope (last two rows): {} params / layer (expected {})",
            slope, r_last.params_per_layer
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "hidden": hidden,
        "vocab": vocab,
        "rows": rows.iter().map(|r| json!({
            "n_layers": r.n_layers,
            "params_total": r.params_total,
            "params_per_layer": r.params_per_layer,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("flow-depth-sweep.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_zero_equals_embedding() {
        let hidden = 128;
        let vocab = 1000;
        assert_eq!(total_params(0, hidden, vocab), vocab * hidden);
    }

    #[test]
    fn params_scale_linearly_with_depth() {
        let hidden = 128;
        let vocab = 1000;
        let a = total_params(1, hidden, vocab);
        let b = total_params(2, hidden, vocab);
        let per = params_per_layer(hidden);
        assert_eq!(b - a, per);
    }

    #[test]
    fn per_layer_formula_matches_expectation() {
        let hidden = 10;
        // 4*100 + 8*100 + 4*10 = 400 + 800 + 40 = 1240
        assert_eq!(params_per_layer(hidden), 1240);
    }

    #[test]
    fn sweep_length_matches_input() {
        let rows = sweep(&[1, 2, 3], 64, 100);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn sweep_monotone_increasing() {
        let rows = sweep(&[1, 2, 4, 8], 64, 100);
        for w in rows.windows(2) {
            assert!(w[1].params_total > w[0].params_total);
        }
    }
}
