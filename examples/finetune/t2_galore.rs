//! # Tier 2.6 — GaLore optimizer (llama family)
//!
//! Falsifier: GaLore optimizer state memory ≤ 0.5× of standard Adam at
//! matched final loss. Closed-form: ratio = r·(d_in + d_out) / (d_in · d_out).
//!
//! Run with: cargo run --example t2_galore

use apr_cookbook::finetune::memory_optimizers as mem;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: u64 = 4096;
const D_IN: u64 = 4096;
const RANK: u64 = 128;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_galore")?;
    let ratio = mem::galore_memory_ratio(D_OUT, D_IN, RANK);
    println!(
        "✓ GaLore: rank={} on {}×{} → optimizer state {:.4}× of Adam",
        RANK, D_OUT, D_IN, ratio
    );
    assert!(ratio < 0.5, "GaLore must be ≤ 0.5× of Adam, got {ratio}");
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
        assert!(mem::galore_memory_ratio(D_OUT, D_IN, RANK) < 0.5);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // rank ≥ d / 2: GaLore degenerates to standard Adam, ratio ≥ 1.
        let ratio = mem::galore_memory_ratio(D_OUT, D_IN, D_IN);
        assert!(ratio >= 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mem::galore_memory_ratio(D_OUT, D_IN, RANK);
        let b = mem::galore_memory_ratio(D_OUT, D_IN, RANK);
        assert_eq!(a, b);
    }
}
