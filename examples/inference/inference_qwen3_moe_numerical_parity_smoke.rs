//! # Inference — Qwen3-MoE Numerical Parity Smoke
//!
//! aprender's Qwen3-MoE numerical-parity bundle (PR #1228) fixed 4
//! root-cause bugs (Q/K RMSNorm rank-3 reshape, rope_theta default rank-4,
//! chat template emission, traced sync) that produced gibberish on
//! Qwen3-Coder-30B-A3B. This recipe is a smoke test for the rank-3
//! RMSNorm reshape — the easiest of the four bugs to demonstrate
//! arithmetically — using synthetic 8-element tensors.
//!
//! Demonstrates the **INF+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PR #1228 + Zhang & Sennrich (2019). Root Mean Square Layer Normalization. NeurIPS. arXiv:1910.07467
//!
//! Run with: cargo run --example inference_qwen3_moe_numerical_parity_smoke
//!
//! Added by PMAT-085 (expand-cookbooks: Tier 3 perf benches).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const EPS: f32 = 1e-6;

/// Rank-3 RMSNorm: input shape [batch, seq, hidden]; normalize across hidden dim.
/// The pre-fix bug was reshape-flatten across all rank-3 dims, which mixed
/// values from different (batch, seq) positions.
fn rms_norm_rank3(input: &[Vec<Vec<f32>>], weight: &[f32]) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|seq| {
            seq.iter()
                .map(|hidden| {
                    let n = hidden.len() as f32;
                    let mean_sq: f32 = hidden.iter().map(|x| x.powi(2)).sum::<f32>() / n;
                    let rrms = (mean_sq + EPS).sqrt().recip();
                    hidden
                        .iter()
                        .zip(weight)
                        .map(|(x, w)| x * rrms * w)
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_qwen3_moe_numerical_parity_smoke")?;

    // Synthetic input: batch=2, seq=2, hidden=4
    let input: Vec<Vec<Vec<f32>>> = vec![
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        vec![vec![1.5, 2.5, 3.5, 4.5], vec![5.5, 6.5, 7.5, 8.5]],
    ];
    let weight: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let out = rms_norm_rank3(&input, &weight);
    println!(
        "rank-3 RMSNorm output shape: [{}, {}, {}]",
        out.len(),
        out[0].len(),
        out[0][0].len()
    );
    println!("first position [0][0]: {:?}", out[0][0]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn rms_norm_per_position_is_independent() {
        // The pre-fix bug mixed values across (batch, seq) positions. Verify
        // that scaling one position doesn't affect another.
        let mut input = vec![vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]]];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let out_a = rms_norm_rank3(&input, &weight);
        // Scale position [0][1] by 100×; position [0][0] should be unchanged.
        for x in &mut input[0][1] {
            *x *= 100.0;
        }
        let out_b = rms_norm_rank3(&input, &weight);
        assert_eq!(out_a[0][0], out_b[0][0]);
    }

    #[test]
    fn output_shape_matches_input_shape() {
        let input = vec![vec![vec![1.0, 2.0, 3.0, 4.0]]];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let out = rms_norm_rank3(&input, &weight);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].len(), 4);
    }
}
