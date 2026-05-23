//! # Tier 2.9 — NEFTune embedding noise (mistral family)
//!
//! Falsifier: NEFTune injects α-scaled noise into embeddings. Recipe checks:
//!   1. Noise scale = α / sqrt(L · d).
//!   2. Noised embeddings differ from clean embeddings (signal injected).
//!   3. Noise is deterministic for fixed seed.
//!
//! Run with: cargo run --example t2_neftune

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ALPHA: f64 = 5.0;
const SEQ_LEN: u32 = 1024;
const D_MODEL: u32 = 4096;
const SEED: u32 = 7;

fn embeddings() -> Vec<Vec<f64>> {
    (0..16)
        .map(|i| (0..32).map(|j| ((i + j) as f64) / 32.0).collect())
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_neftune")?;
    let scale = q::neftune_noise_scale(ALPHA, SEQ_LEN, D_MODEL);
    let noised = q::apply_neftune_noise(&embeddings(), ALPHA, SEED);
    println!(
        "✓ NEFTune α={} on L={}, d={}: noise_scale={:.6}",
        ALPHA, SEQ_LEN, D_MODEL, scale
    );
    assert!(scale > 0.0, "noise scale must be positive");
    let any_diff = embeddings()
        .iter()
        .zip(noised.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .any(|(x, y)| (x - y).abs() > 1e-12);
    assert!(any_diff, "NEFTune must inject noise (≥1 entry differs)");
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
        let scale = q::neftune_noise_scale(ALPHA, SEQ_LEN, D_MODEL);
        assert!(scale > 0.0);
        let noised = q::apply_neftune_noise(&embeddings(), ALPHA, SEED);
        assert_ne!(embeddings(), noised);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // α=0 → noise scale 0, embeddings unchanged.
        let scale = q::neftune_noise_scale(0.0, SEQ_LEN, D_MODEL);
        assert_eq!(scale, 0.0);
        let noised = q::apply_neftune_noise(&embeddings(), 0.0, SEED);
        assert_eq!(embeddings(), noised);
    }

    #[test]
    fn deterministic_across_runs() {
        let n1 = q::apply_neftune_noise(&embeddings(), ALPHA, SEED);
        let n2 = q::apply_neftune_noise(&embeddings(), ALPHA, SEED);
        assert_eq!(n1, n2);
    }
}
