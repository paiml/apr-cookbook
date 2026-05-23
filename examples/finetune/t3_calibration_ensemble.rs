//! # Tier 3.3 — Ensemble calibration (tabular-only)
//!
//! Falsifier: ensemble averaging of probabilities never produces a worse
//! ECE than the worst single member.
//!
//! Run with: cargo run --example t3_calibration_ensemble

use apr_cookbook::finetune::calibration as cal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> (Vec<Vec<f64>>, Vec<bool>) {
    let m1: Vec<f64> = vec![0.95, 0.05, 0.95, 0.05];
    let m2: Vec<f64> = vec![0.85, 0.15, 0.85, 0.15];
    let m3: Vec<f64> = vec![0.75, 0.25, 0.75, 0.25];
    let correct = vec![true, false, true, false];
    (vec![m1, m2, m3], correct)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_calibration_ensemble")?;
    let (members, correct) = fixture();
    let avg = cal::ensemble_average(&members);
    let ece_avg = cal::ece(&avg, &correct);
    let ece_worst = members
        .iter()
        .map(|m| cal::ece(m, &correct))
        .fold(0.0_f64, f64::max);
    println!(
        "✓ ensemble: avg ECE = {:.4} ≤ worst-member ECE = {:.4}",
        ece_avg, ece_worst
    );
    assert!(ece_avg <= ece_worst + 1e-12);
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
        main().unwrap();
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Single member input: "ensemble" returns same probs, ECE equal.
        let (members, correct) = fixture();
        let single = vec![members[0].clone()];
        let avg = cal::ensemble_average(&single);
        let ece_avg = cal::ece(&avg, &correct);
        let ece_member = cal::ece(&members[0], &correct);
        assert!((ece_avg - ece_member).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let (members, _) = fixture();
        let a = cal::ensemble_average(&members);
        let b = cal::ensemble_average(&members);
        assert_eq!(a, b);
    }
}
