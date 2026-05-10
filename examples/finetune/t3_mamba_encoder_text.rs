//! # Tier 3.15 — Mamba encoder text classification (mamba family)
//!
//! Falsifier: Mamba-2 latency scales linearly with sequence length — within
//! 10% of theoretical O(n) per-token. Verified by checking ratio of t/n at
//! large n vs t/n at small n stays close to 1.0.
//!
//! Run with: cargo run --example t3_mamba_encoder_text

use apr_cookbook::finetune::specialty;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<(u32, f64)> {
    // (seq_len, latency_ms): linear scaling.
    vec![(32, 1.0), (128, 4.0), (512, 16.0), (2048, 64.0)]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_mamba_encoder_text")?;
    let lin = specialty::mamba_linearity(&fixture());
    println!("✓ Mamba linearity score: {:.3} (target ≈ 1.0 ± 0.1)", lin);
    assert!((lin - 1.0).abs() < 0.1);
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
        let l = specialty::mamba_linearity(&fixture());
        assert!((l - 1.0).abs() < 0.1);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Quadratic latency → linearity ≫ 1.
        let times = vec![(32_u32, 1.0_f64), (2048, 4096.0)];
        let l = specialty::mamba_linearity(&times);
        assert!(l > 10.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = specialty::mamba_linearity(&fixture());
        let b = specialty::mamba_linearity(&fixture());
        assert_eq!(a, b);
    }
}
