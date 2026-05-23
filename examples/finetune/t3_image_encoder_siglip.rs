//! # Tier 3.9 — SigLIP cosine-similarity argmax (tabular-only)
//!
//! Falsifier: SigLIP cosine-similarity matrix on aligned (text, image) pairs
//! has its row argmax on the diagonal (text_i pairs with image_i).
//!
//! Run with: cargo run --example t3_image_encoder_siglip

use apr_cookbook::finetune::encoders_optimizers as enc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn text_emb() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ]
}
fn image_emb() -> Vec<Vec<f64>> {
    vec![
        vec![0.95, 0.05, 0.0, 0.0],
        vec![0.05, 0.95, 0.0, 0.0],
        vec![0.0, 0.05, 0.95, 0.0],
        vec![0.0, 0.0, 0.05, 0.95],
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_image_encoder_siglip")?;
    let sim = enc::cosine_sim_matrix(&text_emb(), &image_emb());
    println!("✓ SigLIP sim matrix:");
    for row in &sim {
        println!(
            "    {:?}",
            row.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>()
        );
    }
    assert!(enc::diagonal_is_argmax(&sim));
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
        let sim = enc::cosine_sim_matrix(&text_emb(), &image_emb());
        assert!(enc::diagonal_is_argmax(&sim));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Shuffle image embeddings — diagonal is no longer argmax.
        let mut img = image_emb();
        img.swap(0, 1);
        let sim = enc::cosine_sim_matrix(&text_emb(), &img);
        assert!(!enc::diagonal_is_argmax(&sim));
    }

    #[test]
    fn deterministic_across_runs() {
        let s1 = enc::cosine_sim_matrix(&text_emb(), &image_emb());
        let s2 = enc::cosine_sim_matrix(&text_emb(), &image_emb());
        assert_eq!(s1, s2);
    }
}
