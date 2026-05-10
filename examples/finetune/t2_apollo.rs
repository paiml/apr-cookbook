//! # Tier 2.6 — Apollo low-memory optimizer (phi family)
//!
//! Falsifier: Apollo memory ratio ≤ 0.6× of AdamW. Closed-form: ratio =
//! (4 + 4·r·(d_out+d_in)/(d_out·d_in)) / 8 ≈ 0.5 + r/d for d ≫ r.
//!
//! Run with: cargo run --example t2_apollo

use apr_cookbook::finetune::memory_optimizers as mem;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: u64 = 4096;
const D_IN: u64 = 4096;
const RANK: u64 = 64;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_apollo")?;
    let ratio = mem::apollo_memory_ratio(D_OUT, D_IN, RANK);
    println!(
        "✓ Apollo: rank={} on {}×{} → {:.4}× of AdamW",
        RANK, D_OUT, D_IN, ratio
    );
    assert!(ratio < 0.6, "Apollo must be ≤ 0.6× of AdamW, got {ratio}");
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
        assert!(mem::apollo_memory_ratio(D_OUT, D_IN, RANK) < 0.6);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // High rank: ratio approaches 1.0 (no compression benefit).
        let ratio = mem::apollo_memory_ratio(D_OUT, D_IN, D_IN);
        assert!(ratio >= 0.6);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mem::apollo_memory_ratio(D_OUT, D_IN, RANK);
        let b = mem::apollo_memory_ratio(D_OUT, D_IN, RANK);
        assert_eq!(a, b);
    }
}
