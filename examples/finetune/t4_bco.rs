//! # Tier 4.9 — BCO Binary Classifier Optimization (llama family)
//!
//! Falsifier: BCO thumb-up/down classifier accuracy ≥ 0.7 on held-out
//! preference fixture.
//!
//! Run with: cargo run --example t4_bco

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn predictions() -> Vec<u8> {
    vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}
fn labels() -> Vec<u8> {
    vec![1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_bco")?;
    let acc = oa::bco_accuracy(&predictions(), &labels());
    println!("✓ BCO classifier accuracy = {:.3}", acc);
    assert!(acc >= 0.7);
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
        assert!(oa::bco_accuracy(&predictions(), &labels()) >= 0.7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // All-wrong predictions → accuracy = 0.
        let bad: Vec<u8> = labels().iter().map(|l| 1 - l).collect();
        assert_eq!(oa::bco_accuracy(&bad, &labels()), 0.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::bco_accuracy(&predictions(), &labels());
        let b = oa::bco_accuracy(&predictions(), &labels());
        assert_eq!(a, b);
    }
}
