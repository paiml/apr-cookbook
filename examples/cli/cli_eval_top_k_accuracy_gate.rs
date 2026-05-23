//! # apr eval --metric top_k_accuracy — Threshold Gate
//!
//! `apr eval --metric top_k_accuracy --k <N>` reports the fraction of
//! test items where the true label is in the top-K predictions. This
//! recipe builds the metric + a CI gate that promotes/blocks based on
//! a configurable threshold.
//!
//! Demonstrates the **EVAL.5** recipe for PMAT-112 (apr eval coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EVAL-001 + Russakovsky et al. 2015 (ImageNet top-5)
//!
//! Run with: cargo run --example cli_eval_top_k_accuracy_gate
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GateVerdict {
    Pass,
    Fail { observed: f64, threshold: f64 },
    InvalidK,
    EmptyDataset,
}

pub fn top_k_accuracy(predictions: &[Vec<u32>], labels: &[u32], k: usize) -> Option<f64> {
    if predictions.is_empty() || predictions.len() != labels.len() {
        return None;
    }
    if k == 0 {
        return None;
    }
    let hits = predictions
        .iter()
        .zip(labels)
        .filter(|(preds, label)| preds.iter().take(k).any(|p| p == *label))
        .count();
    Some(hits as f64 / predictions.len() as f64)
}

pub fn gate(predictions: &[Vec<u32>], labels: &[u32], k: usize, threshold: f64) -> GateVerdict {
    if k == 0 {
        return GateVerdict::InvalidK;
    }
    let Some(acc) = top_k_accuracy(predictions, labels, k) else {
        return GateVerdict::EmptyDataset;
    };
    if acc >= threshold {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail {
            observed: acc,
            threshold,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_eval_top_k_accuracy_gate")?;

    let preds = vec![
        vec![3u32, 1, 7, 2, 5], // label = 1 (top-2 hit)
        vec![5, 9, 2, 1, 4],    // label = 0 (miss)
        vec![1, 0, 7, 3, 8],    // label = 0 (top-2 hit)
    ];
    let labels = vec![1u32, 0, 0];
    println!("top-1: {:?}", top_k_accuracy(&preds, &labels, 1));
    println!("top-3: {:?}", top_k_accuracy(&preds, &labels, 3));
    println!("gate@0.5: {:?}", gate(&preds, &labels, 3, 0.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perfect_top_1_yields_one() {
        let p = vec![vec![1u32, 2], vec![3, 4]];
        let l = vec![1u32, 3];
        assert_eq!(top_k_accuracy(&p, &l, 1), Some(1.0));
    }

    #[test]
    fn no_label_in_top_k_yields_zero() {
        let p = vec![vec![1u32, 2], vec![3, 4]];
        let l = vec![5u32, 6];
        assert_eq!(top_k_accuracy(&p, &l, 2), Some(0.0));
    }

    #[test]
    fn top_k_widens_recall() {
        let p = vec![vec![3u32, 1, 7, 2, 5]];
        let l = vec![1u32];
        // Label 1 is at position 1 → top-1 misses, top-2 hits.
        assert_eq!(top_k_accuracy(&p, &l, 1), Some(0.0));
        assert_eq!(top_k_accuracy(&p, &l, 2), Some(1.0));
    }

    #[test]
    fn k_zero_returns_none() {
        let p = vec![vec![1u32]];
        let l = vec![1u32];
        assert!(top_k_accuracy(&p, &l, 0).is_none());
    }

    #[test]
    fn mismatched_lengths_returns_none() {
        let p = vec![vec![1u32]];
        let l = vec![1u32, 2];
        assert!(top_k_accuracy(&p, &l, 1).is_none());
    }

    #[test]
    fn gate_invalid_k_returns_invalid_k() {
        assert_eq!(gate(&[], &[], 0, 0.5), GateVerdict::InvalidK);
    }

    #[test]
    fn gate_empty_dataset_returns_empty() {
        assert_eq!(gate(&[], &[], 1, 0.5), GateVerdict::EmptyDataset);
    }

    #[test]
    fn gate_passes_at_threshold_boundary() {
        // ≥ threshold is Pass (inclusive boundary).
        let p = vec![vec![1u32]];
        let l = vec![1u32];
        assert_eq!(gate(&p, &l, 1, 1.0), GateVerdict::Pass);
    }
}
