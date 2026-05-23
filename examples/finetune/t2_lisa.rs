//! # Tier 2.8 — LISA layer-importance sampling (llama family)
//!
//! Falsifier: LISA selects the top-k most important layers per step;
//! all other layers have gradient zero. Recipe checks the mask exactly
//! matches `lisa_select_top_k`.
//!
//! Run with: cargo run --example t2_lisa

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn importances() -> Vec<f64> {
    vec![0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6]
}

const TOP_K: usize = 3;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_lisa")?;
    let mask = q::lisa_gradient_mask(&importances(), TOP_K);
    let active: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();
    println!(
        "✓ LISA top-{} sampled layers: {:?} ({} bottom layers gradient-zero)",
        TOP_K,
        active,
        mask.len() - TOP_K
    );
    assert_eq!(active.len(), TOP_K);
    // Top-3 by importance: indices 1 (0.9), 5 (0.8), 3 (0.7).
    assert!(active.contains(&1));
    assert!(active.contains(&5));
    assert!(active.contains(&3));
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
        let mask = q::lisa_gradient_mask(&importances(), TOP_K);
        let active: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
        assert_eq!(active.len(), TOP_K);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // k=0 → no layer is active, all gradient-zero.
        let mask = q::lisa_gradient_mask(&importances(), 0);
        assert!(mask.iter().all(|&v| !v));
    }

    #[test]
    fn deterministic_across_runs() {
        let m1 = q::lisa_gradient_mask(&importances(), TOP_K);
        let m2 = q::lisa_gradient_mask(&importances(), TOP_K);
        assert_eq!(m1, m2);
    }
}
