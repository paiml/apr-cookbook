//! # Tier 2.6 — Freeze-tuning N layers (gemma family)
//!
//! Falsifier: Freeze-tuning N layers — gradient norms zero on frozen layers,
//! non-zero on trainable. Falsifier checks the mask exactly.
//!
//! Run with: cargo run --example t2_freeze_tuning

use apr_cookbook::finetune::memory_optimizers as mem;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn layer_names() -> Vec<&'static str> {
    vec![
        "embed.weight",
        "layer.0.weight",
        "layer.1.weight",
        "layer.2.weight",
        "layer.3.weight",
        "head.weight",
    ]
}

const FREEZE_PREFIXES: &[&str] = &["embed.", "layer.0.", "layer.1."];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_freeze_tuning")?;
    let names = layer_names();
    let mask = mem::freeze_mask(&names, FREEZE_PREFIXES);
    let grads: Vec<f64> = (0..names.len()).map(|i| 0.1 + i as f64).collect();
    let masked = mem::apply_freeze_mask(&grads, &mask);
    println!(
        "✓ freeze_tuning: {} layers, frozen={:?}, masked_grads={:?}",
        names.len(),
        mask,
        masked
    );
    for (i, &is_frozen) in mask.iter().enumerate() {
        if is_frozen {
            assert_eq!(masked[i], 0.0, "frozen gradient must be zero at i={i}");
        } else {
            assert_ne!(
                masked[i], 0.0,
                "trainable gradient must be non-zero at i={i}"
            );
        }
    }
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
        let names = layer_names();
        let mask = mem::freeze_mask(&names, FREEZE_PREFIXES);
        let frozen = mask.iter().filter(|&&v| v).count();
        // embed., layer.0., layer.1. → 3 frozen.
        assert_eq!(frozen, 3);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // No freeze prefixes → mask is all-false → "frozen has zero grads"
        // is vacuously true, but checking via "trainable_count == n" works.
        let names = layer_names();
        let mask = mem::freeze_mask(&names, &[]);
        assert!(mask.iter().all(|&v| !v));
    }

    #[test]
    fn deterministic_across_runs() {
        let names = layer_names();
        let m1 = mem::freeze_mask(&names, FREEZE_PREFIXES);
        let m2 = mem::freeze_mask(&names, FREEZE_PREFIXES);
        assert_eq!(m1, m2);
    }
}
