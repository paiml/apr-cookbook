//! # Tier 3.5 — Multimodal text+tabular fusion (cross-family)
//!
//! Falsifier: text+tabular gated fusion at gate=0 returns tabular features
//! (modality routing works); at gate→∞ returns text features.
//!
//! Run with: cargo run --example t3_multimodal_text_tabular

use apr_cookbook::finetune::multimodal as mm;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn text() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0]
}
fn tabular() -> Vec<f64> {
    vec![10.0, 20.0, 30.0, 40.0]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_multimodal_text_tabular")?;
    let to_tabular = mm::fuse_gated(&text(), &tabular(), -10.0);
    let to_text = mm::fuse_gated(&text(), &tabular(), 10.0);
    println!(
        "✓ gated fusion: gate=-10 → {:?} ≈ tabular, gate=+10 → {:?} ≈ text",
        to_tabular, to_text
    );
    for (a, b) in to_tabular.iter().zip(tabular().iter()) {
        assert!((a - b).abs() < 0.01);
    }
    for (a, b) in to_text.iter().zip(text().iter()) {
        assert!((a - b).abs() < 0.01);
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
        let f = mm::fuse_gated(&text(), &tabular(), -10.0);
        for (a, b) in f.iter().zip(tabular().iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Mismatched lengths return empty.
        let f = mm::fuse_gated(&[1.0, 2.0], &[3.0], 0.0);
        assert!(f.is_empty());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mm::fuse_gated(&text(), &tabular(), 0.0);
        let b = mm::fuse_gated(&text(), &tabular(), 0.0);
        assert_eq!(a, b);
    }
}
