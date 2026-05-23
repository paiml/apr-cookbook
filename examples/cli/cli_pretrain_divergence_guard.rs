//! # apr pretrain — Divergence Guard (GATE-TRAIN-005/-007/-008)
//!
//! `apr pretrain` exercises three guard rails: GATE-TRAIN-005 (loss
//! must monotonically decrease over a 100-step window), GATE-TRAIN-007
//! (NaN/inf in loss aborts immediately), GATE-TRAIN-008 (gradient norm
//! must stay finite). This recipe builds the per-step guard evaluator
//! as a pure function so a CI pipeline can preview which guard would
//! fire on a given loss trajectory.
//!
//! Demonstrates the **PRETRAIN.4** recipe for PMAT-104 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-TWO-001 + contracts/training-loop-pretrain-v1.yaml
//!
//! Run with: cargo run --example cli_pretrain_divergence_guard
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum GuardVerdict {
    Pass,
    NaNLossAt {
        step: u32,
    },
    NaNGradAt {
        step: u32,
    },
    StagnantLoss {
        window_start: u32,
        window_end: u32,
        mean: f64,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    pub step: u32,
    pub loss: f64,
    pub grad_norm: f64,
}

const MONOTONIC_WINDOW: usize = 100;
const STAGNATION_TOLERANCE: f64 = 0.001;

pub fn evaluate_guards(records: &[StepRecord]) -> GuardVerdict {
    for r in records {
        if !r.loss.is_finite() {
            return GuardVerdict::NaNLossAt { step: r.step };
        }
        if !r.grad_norm.is_finite() {
            return GuardVerdict::NaNGradAt { step: r.step };
        }
    }
    if records.len() >= MONOTONIC_WINDOW {
        for window_start in 0..=(records.len() - MONOTONIC_WINDOW) {
            let window = &records[window_start..window_start + MONOTONIC_WINDOW];
            let mean = window.iter().map(|r| r.loss).sum::<f64>() / window.len() as f64;
            let first_half_mean = window[..MONOTONIC_WINDOW / 2]
                .iter()
                .map(|r| r.loss)
                .sum::<f64>()
                / (MONOTONIC_WINDOW / 2) as f64;
            let second_half_mean = window[MONOTONIC_WINDOW / 2..]
                .iter()
                .map(|r| r.loss)
                .sum::<f64>()
                / (MONOTONIC_WINDOW / 2) as f64;
            if (first_half_mean - second_half_mean).abs() < STAGNATION_TOLERANCE {
                return GuardVerdict::StagnantLoss {
                    window_start: window[0].step,
                    window_end: window[window.len() - 1].step,
                    mean,
                };
            }
        }
    }
    GuardVerdict::Pass
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_divergence_guard")?;

    let healthy: Vec<StepRecord> = (0..120)
        .map(|i| StepRecord {
            step: i,
            loss: 5.0 - (i as f64 * 0.03),
            grad_norm: 0.5,
        })
        .collect();
    println!("healthy:   {:?}", evaluate_guards(&healthy));

    let mut nan_loss = healthy.clone();
    nan_loss[42].loss = f64::NAN;
    println!("nan loss:  {:?}", evaluate_guards(&nan_loss));

    let stagnant: Vec<StepRecord> = (0..120)
        .map(|i| StepRecord {
            step: i,
            loss: 3.0,
            grad_norm: 0.5,
        })
        .collect();
    println!("stagnant:  {:?}", evaluate_guards(&stagnant));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_run(n: u32) -> Vec<StepRecord> {
        (0..n)
            .map(|i| StepRecord {
                step: i,
                loss: 5.0 - (f64::from(i) * 0.01),
                grad_norm: 0.5,
            })
            .collect()
    }

    #[test]
    fn guards_run() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthy_run_passes() {
        assert_eq!(evaluate_guards(&healthy_run(150)), GuardVerdict::Pass);
    }

    #[test]
    fn nan_loss_short_circuits_with_step() {
        let mut r = healthy_run(50);
        r[10].loss = f64::NAN;
        let v = evaluate_guards(&r);
        assert_eq!(v, GuardVerdict::NaNLossAt { step: 10 });
    }

    #[test]
    fn nan_grad_short_circuits_with_step() {
        let mut r = healthy_run(50);
        r[20].grad_norm = f64::NAN;
        let v = evaluate_guards(&r);
        assert_eq!(v, GuardVerdict::NaNGradAt { step: 20 });
    }

    #[test]
    fn inf_loss_treated_as_nan() {
        let mut r = healthy_run(50);
        r[5].loss = f64::INFINITY;
        let v = evaluate_guards(&r);
        assert!(matches!(v, GuardVerdict::NaNLossAt { step: 5 }));
    }

    #[test]
    fn stagnant_loss_flagged_after_window() {
        let stagnant: Vec<StepRecord> = (0..MONOTONIC_WINDOW as u32 + 5)
            .map(|i| StepRecord {
                step: i,
                loss: 3.0,
                grad_norm: 0.5,
            })
            .collect();
        let v = evaluate_guards(&stagnant);
        assert!(matches!(v, GuardVerdict::StagnantLoss { .. }));
    }

    #[test]
    fn shorter_than_window_passes_vacuously() {
        // Without a full window, can't check stagnation — pass.
        let r = healthy_run(50);
        assert_eq!(evaluate_guards(&r), GuardVerdict::Pass);
    }
}
