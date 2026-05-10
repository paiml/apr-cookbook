//! # Tier 3.5 — Multimodal text+image fusion (gemma family)
//!
//! Falsifier: text+image fusion via concat produces a feature vector with
//! dim = d_text + d_image (modalities preserved, not collapsed).
//!
//! Run with: cargo run --example t3_multimodal_text_image

use apr_cookbook::finetune::multimodal as mm;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn text_feat() -> Vec<f64> {
    (0..128).map(|i| (i as f64) / 128.0).collect()
}
fn image_feat() -> Vec<f64> {
    (0..256).map(|i| ((i as f64) - 128.0) / 256.0).collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_multimodal_text_image")?;
    let text = text_feat();
    let image = image_feat();
    let fused = mm::fuse_concat(&text, &image);
    println!(
        "✓ text+image fusion: {} text + {} image → {} fused",
        text.len(),
        image.len(),
        fused.len()
    );
    assert_eq!(fused.len(), text.len() + image.len());
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
        let fused = mm::fuse_concat(&text_feat(), &image_feat());
        assert_eq!(fused.len(), text_feat().len() + image_feat().len());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Single-modal — no fusion happens.
        let single = mm::fuse_concat(&text_feat(), &[]);
        assert_eq!(single.len(), text_feat().len());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mm::fuse_concat(&text_feat(), &image_feat());
        let b = mm::fuse_concat(&text_feat(), &image_feat());
        assert_eq!(a, b);
    }
}
