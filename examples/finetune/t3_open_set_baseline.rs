//! # Tier 3.7 — Open-set baseline (max-softmax) (tabular-only)
//!
//! Falsifier: max-softmax OSR score flags unseen-class samples with
//! AUROC ≥ 0.7 on a held-out fixture.
//!
//! Run with: cargo run --example t3_open_set_baseline

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn unseen_softmax_scores() -> Vec<f64> {
    // OSR score (= 1 − max softmax) for unseen classes — higher.
    let unseen_softmaxes = [
        vec![0.4, 0.3, 0.2, 0.1],
        vec![0.35, 0.3, 0.2, 0.15],
        vec![0.45, 0.25, 0.2, 0.1],
    ];
    unseen_softmaxes
        .iter()
        .map(|s| aou::osr_max_softmax_score(s))
        .collect()
}
fn seen_softmax_scores() -> Vec<f64> {
    let seen_softmaxes = [
        vec![0.85, 0.05, 0.05, 0.05],
        vec![0.9, 0.05, 0.03, 0.02],
        vec![0.95, 0.02, 0.02, 0.01],
    ];
    seen_softmaxes
        .iter()
        .map(|s| aou::osr_max_softmax_score(s))
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_open_set_baseline")?;
    let unseen = unseen_softmax_scores();
    let seen = seen_softmax_scores();
    let auroc = aou::auroc(&unseen, &seen);
    println!("✓ open-set baseline: AUROC = {:.3}", auroc);
    assert!(
        auroc >= 0.7,
        "AUROC must be ≥ 0.7 for separable, got {auroc}"
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
        let a = aou::auroc(&unseen_softmax_scores(), &seen_softmax_scores());
        assert!(a >= 0.7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Identical distributions → AUROC ≈ 0.5.
        let same = vec![0.5, 0.5];
        let a = aou::auroc(&same, &same);
        assert!(a < 0.7);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::auroc(&unseen_softmax_scores(), &seen_softmax_scores());
        let b = aou::auroc(&unseen_softmax_scores(), &seen_softmax_scores());
        assert_eq!(a, b);
    }
}
