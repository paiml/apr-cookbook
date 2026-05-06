//! # Distillation Temperature Search Envelope
//!
//! Distillation temperature T softens teacher logits: T=1 = unmodified,
//! T → ∞ = uniform. Search range: T ∈ [1, 20]; common picks: 2, 4, 8.
//! Best-T heuristic: smallest T where student validation accuracy
//! plateaus within ε. This recipe builds the search envelope + picker.
//!
//! Demonstrates the **DISTILL.5** recipe for PMAT-124 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
//!
//! Run with: cargo run --example distill_temperature_search_envelope
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok(Vec<f64>),
    InvalidRange,
    InvalidStep,
}

const MIN_T: f64 = 1.0;
const MAX_T: f64 = 20.0;

pub fn build_grid(start: f64, end: f64, step: f64) -> SearchVerdict {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return SearchVerdict::InvalidRange;
    }
    if !(MIN_T..=MAX_T).contains(&start) || !(MIN_T..=MAX_T).contains(&end) {
        return SearchVerdict::InvalidRange;
    }
    if start > end {
        return SearchVerdict::InvalidRange;
    }
    if step <= 0.0 {
        return SearchVerdict::InvalidStep;
    }
    let mut grid = Vec::new();
    let mut t = start;
    while t <= end + 1e-9 {
        grid.push(t);
        t += step;
    }
    SearchVerdict::Ok(grid)
}

#[derive(Debug, PartialEq)]
pub enum BestTVerdict {
    Picked { temperature: f64 },
    NoPlateau,
    Empty,
}

const PLATEAU_TOLERANCE: f64 = 1e-3;

pub fn pick_best_t(temps: &[f64], accuracies: &[f64]) -> BestTVerdict {
    if temps.is_empty() || accuracies.is_empty() || temps.len() != accuracies.len() {
        return BestTVerdict::Empty;
    }
    let max_acc = accuracies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_acc.is_finite() {
        return BestTVerdict::NoPlateau;
    }
    // Smallest T whose accuracy is within tolerance of max — favours
    // simpler smoothing.
    for (t, acc) in temps.iter().zip(accuracies.iter()) {
        if (max_acc - acc).abs() <= PLATEAU_TOLERANCE {
            return BestTVerdict::Picked { temperature: *t };
        }
    }
    BestTVerdict::NoPlateau
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_temperature_search_envelope")?;

    println!("grid 1..8 step 1: {:?}", build_grid(1.0, 8.0, 1.0));
    println!("grid invalid: {:?}", build_grid(0.5, 8.0, 1.0));
    let temps = [1.0, 2.0, 4.0, 8.0, 16.0];
    let accs = [0.81, 0.84, 0.85, 0.851, 0.849];
    println!("best T: {:?}", pick_best_t(&temps, &accs));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_grid_has_correct_endpoints() {
        if let SearchVerdict::Ok(grid) = build_grid(1.0, 8.0, 1.0) {
            assert_eq!(grid.first(), Some(&1.0));
            assert_eq!(grid.last(), Some(&8.0));
            assert_eq!(grid.len(), 8);
        }
    }

    #[test]
    fn out_of_range_start_rejected() {
        assert_eq!(build_grid(0.5, 8.0, 1.0), SearchVerdict::InvalidRange);
        assert_eq!(build_grid(1.0, 25.0, 1.0), SearchVerdict::InvalidRange);
    }

    #[test]
    fn start_after_end_rejected() {
        assert_eq!(build_grid(8.0, 4.0, 1.0), SearchVerdict::InvalidRange);
    }

    #[test]
    fn zero_or_negative_step_rejected() {
        assert_eq!(build_grid(1.0, 8.0, 0.0), SearchVerdict::InvalidStep);
        assert_eq!(build_grid(1.0, 8.0, -1.0), SearchVerdict::InvalidStep);
    }

    #[test]
    fn nan_inputs_rejected() {
        assert_eq!(build_grid(f64::NAN, 8.0, 1.0), SearchVerdict::InvalidRange);
    }

    #[test]
    fn picker_finds_smallest_within_tolerance() {
        // Accuracies plateau: T=4 is well within 1e-3 of max (gap ≈ 1e-4).
        let temps = [1.0, 2.0, 4.0, 8.0];
        let accs = [0.81, 0.84, 0.85, 0.8501];
        if let BestTVerdict::Picked { temperature } = pick_best_t(&temps, &accs) {
            assert_eq!(temperature, 4.0);
        } else {
            panic!("expected Picked");
        }
    }

    #[test]
    fn picker_empty_inputs_rejected() {
        assert_eq!(pick_best_t(&[], &[]), BestTVerdict::Empty);
    }

    #[test]
    fn picker_mismatched_lengths_rejected() {
        assert_eq!(pick_best_t(&[1.0], &[0.5, 0.6]), BestTVerdict::Empty);
    }

    #[test]
    fn picker_picks_first_when_all_within_tolerance() {
        let temps = [1.0, 2.0, 4.0];
        let accs = [0.85, 0.85, 0.85];
        if let BestTVerdict::Picked { temperature } = pick_best_t(&temps, &accs) {
            assert_eq!(temperature, 1.0);
        }
    }
}
