//! # Tier 2.6 — DoRA: weight = magnitude × direction (qwen3 family)
//!
//! Falsifier: DoRA decomposes ΔW as direction × magnitude with magnitude norm
//! preserved: ‖direction‖₂ = 1, weight = magnitude · direction (round-trip).
//!
//! Run with: cargo run --example t2_dora

use apr_cookbook::finetune::memory_optimizers as mem;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_weight() -> Vec<f64> {
    (0..256_i32)
        .map(|i| (f64::from(i - 128) / 64.0).sin())
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_dora")?;
    let w = fixture_weight();
    let (m, dir) = mem::dora_decompose(&w);
    let r = mem::dora_reconstruct(m, &dir);
    let dir_norm = mem::vec_norm(&dir);
    println!("✓ DoRA: magnitude={:.4} |direction|={:.6}", m, dir_norm);
    assert!(
        (dir_norm - 1.0).abs() < 1e-12,
        "direction must be unit-norm"
    );
    for (a, b) in w.iter().zip(r.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "DoRA reconstruction must match weight"
        );
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
        let w = fixture_weight();
        let (m, d) = mem::dora_decompose(&w);
        assert!((mem::vec_norm(&d) - 1.0).abs() < 1e-12);
        let r = mem::dora_reconstruct(m, &d);
        for (a, b) in w.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Scale magnitude by 2× → reconstruction is twice the original weight.
        let w = fixture_weight();
        let (m, d) = mem::dora_decompose(&w);
        let r = mem::dora_reconstruct(m * 2.0, &d);
        let any_diff = w.iter().zip(r.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "doubled magnitude must change reconstruction");
    }

    #[test]
    fn deterministic_across_runs() {
        let w = fixture_weight();
        let (m1, d1) = mem::dora_decompose(&w);
        let (m2, d2) = mem::dora_decompose(&w);
        assert_eq!(m1, m2);
        assert_eq!(d1, d2);
    }
}
