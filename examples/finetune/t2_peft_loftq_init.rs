//! # Tier 2.5 — PEFT init — LoftQ (mistral family)
//!
//! Falsifier: LoftQ: 4-bit base quantization error compensated by LoRA init;
//! merge round-trip preserves FP16 baseline within tolerance (max abs error
//! per block ≤ 1/14 ≈ 7.1% on absmax-normalized signals; 0.15 absolute on
//! synthetic [-0.5, 0.5] entries).
//!
//! Run with: cargo run --example t2_peft_loftq_init

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D_OUT: usize = 8;
const D_IN: usize = 64;
const RANK: usize = 4;
const SEED: u32 = 17;
const BLOCK_SIZE: u32 = 64;
const TOL: f64 = 0.15;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_peft_loftq_init")?;
    let art = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Loftq, SEED);
    let err = peft::loftq_round_trip_error(&art.base, BLOCK_SIZE);
    println!(
        "✓ LoftQ init: 4-bit round-trip max abs error = {:.4} (tol {})",
        err, TOL
    );
    assert!(
        err < TOL,
        "LoftQ recovery must keep error < {TOL}, got {err}"
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
        let art = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Loftq, SEED);
        let err = peft::loftq_round_trip_error(&art.base, BLOCK_SIZE);
        assert!(err < TOL);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Fixture with absmax 100 and medium-value entries: per-block quant
        // recovery error scales with absmax (≤ absmax/14), so values like 50
        // recover with error ~7, far exceeding the 0.15 tolerance.
        let mut bad = vec![vec![0.0_f64; D_IN]; D_OUT];
        for i in 0..D_OUT {
            bad[i][0] = 100.0;
            for j in 1..D_IN {
                bad[i][j] = 50.0; // huge mid-value
            }
        }
        let err = peft::loftq_round_trip_error(&bad, D_IN as u32);
        assert!(
            err > TOL,
            "huge-magnitude signal must blow {TOL} tol; got {err}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let art1 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Loftq, SEED);
        let art2 = peft::build_lora_with_init(D_OUT, D_IN, RANK, peft::LoraInit::Loftq, SEED);
        assert_eq!(art1.base, art2.base);
    }
}
