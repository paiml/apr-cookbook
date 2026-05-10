//! # Tier 2.5 — PEFT init — EVA (llama family)
//!
//! Falsifier: EVA-init aligns adapter directions with input-activation SVD;
//! calibration pass produces deterministic init for fixed seed.
//!
//! Run with: cargo run --example t2_peft_eva_init

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: usize = 32;
const D_IN: usize = 64;
const RANK: usize = 8;
const SEED: u32 = 13;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_peft_eva_init")?;
    let art1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED);
    let art2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED);
    println!(
        "✓ EVA init: |A|={:.4} delta={:.4} (deterministic across runs)",
        art1.report.a_norm, art1.report.initial_delta_norm
    );
    assert_eq!(
        art1.a, art2.a,
        "EVA init must be deterministic for fixed seed"
    );
    assert_eq!(art1.report.a_norm, art2.report.a_norm);
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
        let a1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED).a;
        let a2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED).a;
        assert_eq!(a1, a2);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let a1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, 13).a;
        let a2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, 17).a;
        assert_ne!(a1, a2);
    }

    #[test]
    fn deterministic_across_runs() {
        let art1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED);
        let art2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Eva, SEED);
        assert_eq!(art1.a, art2.a);
        assert_eq!(art1.b, art2.b);
    }
}
