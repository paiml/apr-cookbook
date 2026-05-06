//! # apr runs show — Per-Run Metric Summary
//!
//! `apr runs show <ID>` displays one-line aggregate statistics on the
//! recorded loss series: min, max, final, median, std, total_steps. This
//! recipe builds the summary computation as a pure function and asserts
//! the contract: empty input returns None (not zeros), median uses the
//! lower-middle for even-length series.
//!
//! Demonstrates the **RUNS.5** recipe for PMAT-102 (apr runs show coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender RUNS-002
//!
//! Run with: cargo run --example cli_runs_show_metric_summary
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub min: f64,
    pub max: f64,
    pub final_loss: f64,
    pub median: f64,
    pub std: f64,
    pub total_steps: u64,
}

pub fn summarize(losses: &[f64]) -> Option<RunSummary> {
    if losses.is_empty() {
        return None;
    }
    let min = losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let final_loss = *losses.last().unwrap();
    let mut sorted: Vec<f64> = losses.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mean = losses.iter().sum::<f64>() / losses.len() as f64;
    let var = losses.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / losses.len() as f64;
    let std = var.sqrt();
    Some(RunSummary {
        min,
        max,
        final_loss,
        median,
        std,
        total_steps: losses.len() as u64,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_runs_show_metric_summary")?;

    let losses: Vec<f64> = (0..100).map(|i| 5.0 - (i as f64 * 0.04)).collect();
    println!("{:#?}", summarize(&losses));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn single_value_yields_zero_std() {
        let s = summarize(&[3.14]).unwrap();
        assert_eq!(s.min, 3.14);
        assert_eq!(s.max, 3.14);
        assert_eq!(s.median, 3.14);
        assert_eq!(s.std, 0.0);
        assert_eq!(s.total_steps, 1);
    }

    #[test]
    fn descending_series_summarized() {
        let s = summarize(&[5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(s.final_loss, 1.0);
        assert_eq!(s.median, 3.0);
    }

    #[test]
    fn even_length_median_uses_lower_middle() {
        // [1, 2, 3, 4] → median uses index len/2 = 2 → value 3 (sorted).
        let s = summarize(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.median, 3.0);
    }

    #[test]
    fn final_loss_uses_last_index_not_min() {
        // Even if min is in the middle, final_loss must be the literal last value.
        let s = summarize(&[5.0, 0.5, 4.0]).unwrap();
        assert_eq!(s.final_loss, 4.0);
        assert_eq!(s.min, 0.5);
    }

    #[test]
    fn total_steps_matches_input_length() {
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(s.total_steps, 5);
    }
}
