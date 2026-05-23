//! # Tier 2.8 — ReLoRA (llama family)
//!
//! Falsifier: ReLoRA periodic merge-and-restart — cumulative effective rank
//! > single-LoRA rank-r over T restarts. Closed-form: cum_rank = min(r·T, d).
//!
//! Run with: cargo run --example t2_relora

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const RANK: u32 = 8;
const RESTARTS: u32 = 4;
const D: u32 = 4096;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_relora")?;
    let cum = q::relora_cumulative_rank(RANK, RESTARTS, D);
    println!(
        "✓ ReLoRA: rank={} × {} restarts on d={} → cumulative rank {}",
        RANK, RESTARTS, D, cum
    );
    assert!(
        cum > RANK,
        "ReLoRA cumulative rank must exceed single-LoRA rank, got {cum}"
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
        let cum = q::relora_cumulative_rank(RANK, RESTARTS, D);
        assert!(cum > RANK);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // 1 restart = single-LoRA rank, no cumulative gain.
        let cum = q::relora_cumulative_rank(RANK, 1, D);
        assert_eq!(cum, RANK);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = q::relora_cumulative_rank(RANK, RESTARTS, D);
        let b = q::relora_cumulative_rank(RANK, RESTARTS, D);
        assert_eq!(a, b);
    }
}
