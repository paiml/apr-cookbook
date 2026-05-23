//! # Tier 2.5 — TinyLoRA — gemma family
//!
//! Falsifier: TinyLoRA at rank 1 has ≤ 0.06% trainable params yet achieves
//! measurable loss decrease on simple SFT.
//!
//! Run with: cargo run --example t2_tinylora

use apr_cookbook::finetune::lora;
use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: u64 = 4096;
const D_IN: u64 = 4096;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_tinylora")?;
    let ratio = peft::tinylora_reduction_ratio(D_IN, D_OUT);
    println!(
        "✓ TinyLoRA r=1 on {}×{}: trainable_ratio={:.6} ({:.4}%)",
        D_OUT,
        D_IN,
        ratio,
        ratio * 100.0
    );
    assert!(
        ratio < 0.0006,
        "TinyLoRA must keep trainable ratio < 0.06%, got {ratio}"
    );
    // Demonstrate that rank-1 LoRA still trains: SGD reduces loss.
    let mut layer = lora::LoraLayer::new(4, 4, 1, 1.0);
    let x = vec![1.0, 0.5, -0.5, 0.2];
    let target = vec![5.0, 4.0, 3.0, 2.0];
    let (init_loss, final_loss, _) = lora::run_smoke_train(&mut layer, &target, &x, 100)?;
    println!(
        "  rank-1 SGD: initial={:.4} final={:.4}",
        init_loss, final_loss
    );
    assert!(
        final_loss < init_loss,
        "TinyLoRA r=1 must still reduce loss"
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
        let ratio = peft::tinylora_reduction_ratio(D_IN, D_OUT);
        assert!(ratio < 0.0006);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Tiny dimensions don't have negligible trainable ratio: at d=4 ratio ≈ 0.33.
        let ratio = peft::tinylora_reduction_ratio(4, 4);
        assert!(ratio > 0.1, "small-d ratio should not be ≤ 0.06%");
    }

    #[test]
    fn deterministic_across_runs() {
        let a = peft::tinylora_reduction_ratio(D_IN, D_OUT);
        let b = peft::tinylora_reduction_ratio(D_IN, D_OUT);
        assert_eq!(a, b);
    }
}
