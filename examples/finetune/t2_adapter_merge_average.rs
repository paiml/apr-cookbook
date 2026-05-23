//! # Tier 2.4 — Adapter merge — average (qwen3 family)
//!
//! Falsifier: average merge of identical LoRAs returns the input unchanged
//! (bit-identical). For non-identical inputs, output = mean of inputs.
//!
//! Run with: cargo run --example t2_adapter_merge_average

use apr_cookbook::finetune::adapter_merge as am;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<f64> {
    (0..32)
        .map(|i| (((i as u32 * 5 + 2) % 13) as f64) / 13.0 - 0.4)
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_adapter_merge_average")?;
    let a = fixture();
    let merged = am::average_merge(&[a.clone(), a.clone(), a.clone()]);
    println!(
        "✓ average merge of 3 identical: |a|={:.4} |merged|={:.4}",
        am::norm(&a),
        am::norm(&merged)
    );
    for (x, y) in a.iter().zip(merged.iter()) {
        assert!(
            (x - y).abs() < 1e-12,
            "average of identicals must equal the input"
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
        let a = fixture();
        let merged = am::average_merge(&[a.clone(), a.clone(), a.clone()]);
        for (x, y) in a.iter().zip(merged.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Averaging an input with its negation should give zero, not the input.
        let a = fixture();
        let neg: Vec<f64> = a.iter().map(|v| -v).collect();
        let merged = am::average_merge(&[a.clone(), neg]);
        for v in &merged {
            assert!(v.abs() < 1e-12, "average of x and -x must be 0");
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = fixture();
        let m1 = am::average_merge(&[a.clone(), a.clone()]);
        let m2 = am::average_merge(&[a.clone(), a]);
        assert_eq!(m1, m2);
    }
}
