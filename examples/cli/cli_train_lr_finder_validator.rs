//! # apr train --lr-finder — Smith-Style LR Range Picker
//!
//! Smith 2017's LR range test trains for ~100 mini-batches with
//! exponentially increasing LR. The "best" LR is one decade below the
//! point of minimum loss. This recipe builds the picker over a
//! (lr, loss) trajectory.
//!
//! Demonstrates the **TRAIN.5** recipe for PMAT-116 (apr train coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TRAIN-001 + Smith 2017 (LR range test)
//!
//! Run with: cargo run --example cli_train_lr_finder_validator
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LrFinderVerdict {
    Ok { suggested_lr: f64 },
    NotEnoughSamples,
    NoMinimumFound,
    InvalidLrSequence,
}

const MIN_SAMPLES: usize = 20;

pub fn pick_lr(lrs: &[f64], losses: &[f64]) -> LrFinderVerdict {
    if lrs.len() != losses.len() || lrs.len() < MIN_SAMPLES {
        return LrFinderVerdict::NotEnoughSamples;
    }
    if !lrs.windows(2).all(|w| w[0] < w[1] && w[0] > 0.0) {
        return LrFinderVerdict::InvalidLrSequence;
    }
    let Some((min_idx, _)) = losses
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return LrFinderVerdict::NoMinimumFound;
    };
    // Step back ~one decade (ratio of 10).
    let target_lr = lrs[min_idx] / 10.0;
    LrFinderVerdict::Ok {
        suggested_lr: target_lr,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_lr_finder_validator")?;

    // Synthetic exponential LR sweep with U-shaped loss.
    let lrs: Vec<f64> = (0..30).map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0)).collect();
    let losses: Vec<f64> = lrs
        .iter()
        .map(|lr| {
            let log_lr = lr.log10();
            (log_lr + 3.0).powi(2) + 0.1
        })
        .collect();
    println!("found: {:?}", pick_lr(&lrs, &losses));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_trajectory_picks_one_decade_below() {
        let lrs: Vec<f64> = (0..30).map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0)).collect();
        let losses: Vec<f64> = lrs
            .iter()
            .map(|lr| {
                let log_lr = lr.log10();
                (log_lr + 3.0).powi(2) + 0.1
            })
            .collect();
        if let LrFinderVerdict::Ok { suggested_lr } = pick_lr(&lrs, &losses) {
            // Min loss is around lr=1e-3 (log_lr = -3); suggest = 1e-4.
            assert!(suggested_lr < 1e-3);
            assert!(suggested_lr > 1e-5);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn too_few_samples_rejected() {
        let lrs = vec![1e-5, 1e-4, 1e-3];
        let losses = vec![3.0, 2.0, 1.0];
        assert_eq!(pick_lr(&lrs, &losses), LrFinderVerdict::NotEnoughSamples);
    }

    #[test]
    fn mismatched_lengths_rejected() {
        let lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-6 * 10f64.powf(i as f64))
            .collect();
        let losses = vec![1.0; MIN_SAMPLES - 1];
        assert_eq!(pick_lr(&lrs, &losses), LrFinderVerdict::NotEnoughSamples);
    }

    #[test]
    fn non_monotonic_lr_rejected() {
        let mut lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0))
            .collect();
        lrs[10] = 1e-9; // out of order
        let losses = vec![1.0; MIN_SAMPLES];
        assert_eq!(pick_lr(&lrs, &losses), LrFinderVerdict::InvalidLrSequence);
    }

    #[test]
    fn zero_lr_rejected() {
        let mut lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0))
            .collect();
        lrs[0] = 0.0;
        let losses = vec![1.0; MIN_SAMPLES];
        assert_eq!(pick_lr(&lrs, &losses), LrFinderVerdict::InvalidLrSequence);
    }

    #[test]
    fn all_nan_losses_no_minimum() {
        let lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0))
            .collect();
        let losses = vec![f64::NAN; MIN_SAMPLES];
        assert_eq!(pick_lr(&lrs, &losses), LrFinderVerdict::NoMinimumFound);
    }

    #[test]
    fn nan_losses_skipped_finite_minimum_used() {
        let lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-6 * 10f64.powf(i as f64 / 5.0))
            .collect();
        let mut losses = vec![f64::NAN; MIN_SAMPLES];
        losses[5] = 0.5;
        losses[10] = 0.1; // minimum
        losses[15] = 0.3;
        if let LrFinderVerdict::Ok { suggested_lr } = pick_lr(&lrs, &losses) {
            assert!((suggested_lr - lrs[10] / 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn suggested_lr_always_smaller_than_min_loss_lr() {
        let lrs: Vec<f64> = (0..MIN_SAMPLES)
            .map(|i| 1e-5 * 10f64.powf(i as f64 / 4.0))
            .collect();
        let losses: Vec<f64> = (0..MIN_SAMPLES).map(|i| (i as f64 - 12.0).abs()).collect();
        if let LrFinderVerdict::Ok { suggested_lr } = pick_lr(&lrs, &losses) {
            // Suggested = min/10 < min.
            assert!(suggested_lr < lrs[12]);
        }
    }
}
