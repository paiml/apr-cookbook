//! # Tier 1.1 — SFT minimal — Phi
//!
//! Single-epoch supervised fine-tuning on a 100-row JSONL fixture using
//! a tiny linear regression head as a stand-in for the LLM training loop.
//! The falsifier asserts loss decreases monotonically over the epoch.
//!
//! Demonstrates the **t1_sft_minimal_phi** recipe per
//! `docs/specifications/fine-tuning-cookbook.md` v1.2.0 (PMAT-331).
//!
//! ## Mirror
//!
//! - Ludwig: examples/getting_started/
//! - Unsloth: nb/Phi3.1_(8B)-Alpaca.ipynb
//! - apr-native: false
//!
//! ## Falsifiable claim
//!
//! Training loss decreases over 1 epoch (100 SFT examples) for Phi-tiny.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/finetune-t1-sft-minimal-phi-v1.yaml
//! Citation: Abdin et al. (2024). Phi-3 Technical Report. arXiv:2404.14219
//!
//! Run with: cargo run --example t1_sft_minimal_phi
//!
//! Added by PMAT-331.

use apr_cookbook::finetune::sft_minimal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FAMILY: &str = "phi";
const FIXTURE: &str = "tests/fixtures/finetune/t1_sft_minimal_phi/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_sft_minimal_phi")?;
    let result = sft_minimal::run(FAMILY, FIXTURE, 42, 1)?;
    println!(
        "✓ {} SFT minimal: loss {:.4} → {:.4} ({} steps, ratio {:.3})",
        FAMILY,
        result.loss_initial,
        result.loss_final,
        result.step_count,
        result.convergence_ratio()
    );
    assert!(
        result.loss_decreased(),
        "falsifier broke: loss did not decrease ({} → {})",
        result.loss_initial,
        result.loss_final
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
        let r = sft_minimal::run(FAMILY, FIXTURE, 42, 1).expect("run");
        assert!(
            r.loss_decreased(),
            "loss should decrease: initial={} final={}",
            r.loss_initial,
            r.loss_final
        );
        assert_eq!(r.family, FAMILY);
        assert_eq!(r.epoch_count, 1);
        assert_eq!(r.step_count, 100);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Zero-epoch training is the perturbation: loss can't decrease
        // because no SGD steps run. Falsifier should detect this.
        let r = sft_minimal::run(FAMILY, FIXTURE, 42, 0).expect("run zero-epoch");
        assert!(
            !r.loss_decreased(),
            "0-epoch run should not decrease loss (initial={} final={})",
            r.loss_initial,
            r.loss_final
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = sft_minimal::run(FAMILY, FIXTURE, 42, 1).expect("a");
        let b = sft_minimal::run(FAMILY, FIXTURE, 42, 1).expect("b");
        assert_eq!(a.loss_final, b.loss_final);
        assert_eq!(a.step_count, b.step_count);
    }
}
