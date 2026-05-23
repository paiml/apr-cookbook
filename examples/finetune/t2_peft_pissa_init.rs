//! # Tier 2.5 — PEFT init — PiSSA (mistral family)
//!
//! Falsifier: PiSSA-init: A,B drawn from base-weight SVD; loss starts below
//! random-init baseline at step 0 because initial delta is non-zero.
//!
//! Run with: cargo run --example t2_peft_pissa_init

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: usize = 32;
const D_IN: usize = 32;
const RANK: usize = 8;
const SEED: u32 = 11;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_peft_pissa_init")?;
    let r_random =
        peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Random, SEED).report;
    let r_pissa = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Pissa, SEED).report;
    println!(
        "✓ PiSSA init: random_delta={:.6} pissa_delta={:.6}",
        r_random.initial_delta_norm, r_pissa.initial_delta_norm
    );
    assert!(r_random.is_zero_init);
    assert!(!r_pissa.is_zero_init && r_pissa.initial_delta_norm > 1e-6);
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
        let r = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Pissa, SEED).report;
        assert!(!r.is_zero_init);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let r = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Random, SEED).report;
        assert!(r.is_zero_init);
    }

    #[test]
    fn deterministic_across_runs() {
        let art1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Pissa, SEED);
        let art2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Pissa, SEED);
        assert_eq!(art1.a, art2.a);
        assert_eq!(art1.b, art2.b);
    }
}
