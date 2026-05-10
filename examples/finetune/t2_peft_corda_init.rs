//! # Tier 2.5 — PEFT init — CorDA (llama family)
//!
//! Falsifier: CorDA init scales A by sqrt(2/d_in), giving smaller A norm
//! than uniform-random init. (Variance-preserving alignment with covariance
//! directions; the "≥20% fewer steps to target loss" downstream property
//! follows from this scaling but is a stochastic claim.)
//!
//! Run with: cargo run --example t2_peft_corda_init

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: usize = 32;
const D_IN: usize = 64;
const RANK: usize = 8;
const SEED: u32 = 7;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_peft_corda_init")?;
    let r_random =
        peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Random, SEED).report;
    let r_corda = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Corda, SEED).report;
    println!(
        "✓ CorDA init: |A_random|={:.4} |A_corda|={:.4} delta_corda={:.4}",
        r_random.a_norm, r_corda.a_norm, r_corda.initial_delta_norm
    );
    assert!(
        r_corda.a_norm < r_random.a_norm,
        "CorDA must scale A smaller than random"
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
        let r_random =
            peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Random, SEED).report;
        let r_corda =
            peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Corda, SEED).report;
        assert!(r_corda.a_norm < r_random.a_norm);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // For very small d_in (e.g., 1), sqrt(2/d_in) > 1, so CorDA's A norm
        // would be LARGER than random — falsifier (corda < random) breaks.
        let r_random = peft::build_lora_with_init(8, 1, 4, peft::LoraInit::Random, 5).report;
        let r_corda = peft::build_lora_with_init(8, 1, 4, peft::LoraInit::Corda, 5).report;
        assert!(r_corda.a_norm >= r_random.a_norm);
    }

    #[test]
    fn deterministic_across_runs() {
        let art1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Corda, SEED);
        let art2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Corda, SEED);
        assert_eq!(art1.a, art2.a);
        assert_eq!(art1.b, art2.b);
    }
}
