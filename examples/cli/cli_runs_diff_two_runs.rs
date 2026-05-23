//! # apr runs diff — Two-Run Side-by-Side Comparison
//!
//! `apr runs diff <RUN_A> <RUN_B>` aligns two loss curves on step index
//! and reports per-step Δloss = a - b. This recipe builds the aligner +
//! diff calculator and asserts the contract: shorter series gets padded
//! with NaN at the tail, NaN values propagate (don't silently zero), the
//! aggregate "winning" verdict picks the run with lower mean.
//!
//! Demonstrates the **RUNS.6** recipe for PMAT-102 (apr runs diff coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender RUNS-003
//!
//! Run with: cargo run --example cli_runs_diff_two_runs
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct StepDiff {
    pub step: usize,
    pub a: f64,
    pub b: f64,
    pub delta: f64, // a - b
}

#[derive(Debug, Clone, PartialEq)]
pub enum WinnerVerdict {
    AWins { delta_mean: f64 },
    BWins { delta_mean: f64 },
    Tie,
    Inconclusive, // contains NaN
}

pub fn align_and_diff(a: &[f64], b: &[f64]) -> Vec<StepDiff> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|step| {
            let av = *a.get(step).unwrap_or(&f64::NAN);
            let bv = *b.get(step).unwrap_or(&f64::NAN);
            StepDiff {
                step,
                a: av,
                b: bv,
                delta: av - bv,
            }
        })
        .collect()
}

pub fn winning_verdict(diffs: &[StepDiff]) -> WinnerVerdict {
    if diffs.is_empty() {
        return WinnerVerdict::Tie;
    }
    let finite_deltas: Vec<f64> = diffs
        .iter()
        .map(|d| d.delta)
        .filter(|d| d.is_finite())
        .collect();
    if finite_deltas.len() < diffs.len() / 2 {
        // Fewer than half the points are finite — too noisy to call.
        return WinnerVerdict::Inconclusive;
    }
    let mean: f64 = finite_deltas.iter().sum::<f64>() / finite_deltas.len() as f64;
    if (mean.abs()) < 1e-9 {
        WinnerVerdict::Tie
    } else if mean < 0.0 {
        // a - b < 0 means a's loss is lower → A wins.
        WinnerVerdict::AWins { delta_mean: mean }
    } else {
        WinnerVerdict::BWins { delta_mean: mean }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_runs_diff_two_runs")?;

    let a: Vec<f64> = (0..10).map(|i| 5.0 - (i as f64 * 0.5)).collect();
    let b: Vec<f64> = (0..10).map(|i| 5.0 - (i as f64 * 0.4)).collect();
    let d = align_and_diff(&a, &b);
    println!("verdict: {:?}", winning_verdict(&d));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_series_diff_to_zero() {
        let d = align_and_diff(&[1.0, 2.0], &[1.0, 2.0]);
        for step in &d {
            assert_eq!(step.delta, 0.0);
        }
    }

    #[test]
    fn shorter_series_padded_with_nan() {
        let d = align_and_diff(&[1.0, 2.0], &[1.0]);
        assert!(d[1].b.is_nan());
        assert!(d[1].delta.is_nan());
    }

    #[test]
    fn a_lower_loss_wins() {
        // a is uniformly 0.1 below b → A wins.
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 3.1];
        let d = align_and_diff(&a, &b);
        let v = winning_verdict(&d);
        assert!(matches!(v, WinnerVerdict::AWins { .. }));
    }

    #[test]
    fn b_lower_loss_wins() {
        let a = vec![1.0, 2.0];
        let b = vec![0.5, 1.5];
        let d = align_and_diff(&a, &b);
        let v = winning_verdict(&d);
        assert!(matches!(v, WinnerVerdict::BWins { .. }));
    }

    #[test]
    fn equal_means_yield_tie() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = align_and_diff(&a, &b);
        assert_eq!(winning_verdict(&d), WinnerVerdict::Tie);
    }

    #[test]
    fn empty_input_returns_tie() {
        assert_eq!(winning_verdict(&[]), WinnerVerdict::Tie);
    }

    #[test]
    fn mostly_nan_returns_inconclusive() {
        let mut a = vec![f64::NAN; 10];
        a[0] = 1.0;
        let b = vec![1.0; 10];
        let d = align_and_diff(&a, &b);
        assert_eq!(winning_verdict(&d), WinnerVerdict::Inconclusive);
    }

    #[test]
    fn step_indices_are_sequential() {
        let d = align_and_diff(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        for (i, step) in d.iter().enumerate() {
            assert_eq!(step.step, i);
        }
    }
}
