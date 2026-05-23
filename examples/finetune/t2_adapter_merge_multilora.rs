//! # Tier 2.4 — Adapter merge — multi-LoRA (gemma family)
//!
//! Falsifier: loading 3 LoRAs simultaneously preserves per-LoRA outputs in
//! a routing test. Concretely: when each LoRA's nonzero coordinates are
//! disjoint (orthogonal), stacking them additively is equivalent to applying
//! each individually.
//!
//! Run with: cargo run --example t2_adapter_merge_multilora

use apr_cookbook::finetune::adapter_merge as am;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn base() -> Vec<f64> {
    vec![0.0; 9]
}

fn lora1() -> Vec<f64> {
    let mut v = vec![0.0; 9];
    v[0] = 0.5;
    v[3] = 0.5;
    v[6] = 0.5;
    v
}

fn lora2() -> Vec<f64> {
    let mut v = vec![0.0; 9];
    v[1] = 0.7;
    v[4] = 0.7;
    v[7] = 0.7;
    v
}

fn lora3() -> Vec<f64> {
    let mut v = vec![0.0; 9];
    v[2] = 0.3;
    v[5] = 0.3;
    v[8] = 0.3;
    v
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_adapter_merge_multilora")?;
    let stacked = am::multilora_apply(&base(), &[lora1(), lora2(), lora3()]);
    let l1_alone = am::multilora_apply(&base(), &[lora1()]);
    let l2_alone = am::multilora_apply(&base(), &[lora2()]);
    let l3_alone = am::multilora_apply(&base(), &[lora3()]);
    println!(
        "✓ multi-LoRA: stacked={:?} expected_sum=lora1+lora2+lora3",
        stacked
    );
    for i in 0..9 {
        let expected = l1_alone[i] + l2_alone[i] + l3_alone[i];
        assert!(
            (stacked[i] - expected).abs() < 1e-12,
            "multi-LoRA at i={i} must equal sum of individual: {} vs {}",
            stacked[i],
            expected
        );
    }
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
        let stacked = am::multilora_apply(&base(), &[lora1(), lora2(), lora3()]);
        for i in 0..9 {
            let expected = lora1()[i] + lora2()[i] + lora3()[i];
            assert!((stacked[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // If we forget LoRA-3, stacked output != l1 + l2 + l3.
        let two_only = am::multilora_apply(&base(), &[lora1(), lora2()]);
        let l3_present = am::multilora_apply(&base(), &[lora1(), lora2(), lora3()]);
        assert_ne!(two_only, l3_present);
    }

    #[test]
    fn deterministic_across_runs() {
        let m1 = am::multilora_apply(&base(), &[lora1(), lora2(), lora3()]);
        let m2 = am::multilora_apply(&base(), &[lora1(), lora2(), lora3()]);
        assert_eq!(m1, m2);
    }
}
