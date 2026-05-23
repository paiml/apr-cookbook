//! # Tier 2.5 — Regex-based layer freeze — phi family
//!
//! Falsifier: regex-based layer freeze produces a gradient mask that is
//! `true` for parameters whose name matches the freeze pattern, so trainable
//! params are exactly the complement.
//!
//! Run with: cargo run --example t2_regex_freeze

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn parameter_names() -> Vec<&'static str> {
    vec![
        "encoder.layer.0.weight",
        "encoder.layer.0.bias",
        "encoder.layer.1.weight",
        "encoder.layer.1.bias",
        "encoder.layer.2.weight",
        "encoder.layer.2.bias",
        "decoder.layer.0.weight",
        "decoder.layer.0.bias",
        "lm_head.weight",
    ]
}

const FREEZE_PATTERN: &str = r"^encoder\.layer\.[01]\.";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_regex_freeze")?;
    let names = parameter_names();
    let mask = peft::regex_freeze_mask(&names, FREEZE_PATTERN)?;
    let frozen = mask.iter().filter(|&&v| v).count();
    let trainable = mask.iter().filter(|&&v| !v).count();
    println!(
        "✓ regex_freeze pattern={:?}: {} frozen, {} trainable",
        FREEZE_PATTERN, frozen, trainable
    );
    // Expect encoder.layer.0.* + encoder.layer.1.* = 4 frozen, 5 trainable.
    assert_eq!(frozen, 4);
    assert_eq!(trainable, 5);
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
        let mask = peft::regex_freeze_mask(&parameter_names(), FREEZE_PATTERN).unwrap();
        let frozen = mask.iter().filter(|&&v| v).count();
        assert_eq!(frozen, 4);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Different pattern produces a different mask.
        let mask = peft::regex_freeze_mask(&parameter_names(), r"^lm_head").unwrap();
        let frozen = mask.iter().filter(|&&v| v).count();
        assert_eq!(frozen, 1);
    }

    #[test]
    fn deterministic_across_runs() {
        let m1 = peft::regex_freeze_mask(&parameter_names(), FREEZE_PATTERN).unwrap();
        let m2 = peft::regex_freeze_mask(&parameter_names(), FREEZE_PATTERN).unwrap();
        assert_eq!(m1, m2);
    }
}
