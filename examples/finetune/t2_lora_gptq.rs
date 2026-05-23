//! # Tier 2.7 — LoRA on GPTQ 4-bit base (phi family)
//!
//! Falsifier: GPTQ 4-bit base + LoRA — per-block reconstruction error
//! ≤ tolerance × σ_block. Closed-form: max relative block error ≤ tol_rel.
//!
//! Run with: cargo run --example t2_lora_gptq

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn blocks() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.5, 0.25, -0.5],
        vec![2.0, -1.0, 0.5, -0.25],
        vec![0.1, 0.05, -0.1, 0.05],
    ]
}

fn errors() -> Vec<Vec<f64>> {
    vec![
        vec![0.05, 0.02, 0.01, -0.03],     // |err|/|block| ~ 0.06/1.21 ~ 0.05
        vec![0.08, -0.04, 0.02, -0.01],    // ~ 0.09/2.30 ~ 0.04
        vec![0.005, 0.002, -0.005, 0.002], // ~ 0.008/0.16 ~ 0.05
    ]
}

const TOL_REL: f64 = 0.07;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_lora_gptq")?;
    let max_rel = q::gptq_max_relative_block_error(&blocks(), &errors());
    println!(
        "✓ GPTQ: max per-block relative error = {:.4} (tol {})",
        max_rel, TOL_REL
    );
    assert!(
        max_rel <= TOL_REL,
        "GPTQ relative error must be ≤ {TOL_REL}, got {max_rel}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let m = q::gptq_max_relative_block_error(&blocks(), &errors());
        assert!(m <= TOL_REL);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Multiply errors by 10× — relative error blows tolerance.
        let big_errors: Vec<Vec<f64>> = errors()
            .iter()
            .map(|b| b.iter().map(|x| x * 10.0).collect())
            .collect();
        let m = q::gptq_max_relative_block_error(&blocks(), &big_errors);
        assert!(m > TOL_REL);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = q::gptq_max_relative_block_error(&blocks(), &errors());
        let b = q::gptq_max_relative_block_error(&blocks(), &errors());
        assert_eq!(a, b);
    }
}
