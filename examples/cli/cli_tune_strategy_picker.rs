//! # apr tune --strategy — HPO Strategy Picker
//!
//! `apr tune --strategy <S>` accepts {grid, random, tpe}. Decision rules:
//! grid is exhaustive (fine for ≤ 50 trials, exponential blowup beyond);
//! random scales linearly; TPE (Tree-structured Parzen Estimator) is the
//! sample-efficient default for ≥ 20 trials with ≥ 4 dimensions. This
//! recipe builds the auto-picker.
//!
//! Demonstrates the **TUNE.7** recipe for PMAT-111 (apr tune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HPO-001 + Bergstra & Bengio 2012 (TPE)
//!
//! Run with: cargo run --example cli_tune_strategy_picker
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpoStrategy {
    Grid,
    Random,
    Tpe,
}

impl HpoStrategy {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "grid" => Some(HpoStrategy::Grid),
            "random" => Some(HpoStrategy::Random),
            "tpe" => Some(HpoStrategy::Tpe),
            _ => None,
        }
    }
}

const GRID_TRIAL_LIMIT: u32 = 50;
const TPE_DIMENSION_FLOOR: usize = 4;
const TPE_TRIAL_FLOOR: u32 = 20;

pub fn auto_pick(num_trials: u32, num_dimensions: usize) -> HpoStrategy {
    if num_dimensions == 0 || num_trials == 0 {
        return HpoStrategy::Random;
    }
    if num_trials <= GRID_TRIAL_LIMIT && num_dimensions <= 2 {
        return HpoStrategy::Grid;
    }
    if num_trials >= TPE_TRIAL_FLOOR && num_dimensions >= TPE_DIMENSION_FLOOR {
        return HpoStrategy::Tpe;
    }
    HpoStrategy::Random
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tune_strategy_picker")?;

    let cases = [
        ("tiny grid", 16u32, 2usize),
        ("medium random", 30, 3),
        ("large complex", 100, 6),
        ("zero trials", 0, 4),
        ("zero dims", 50, 0),
    ];
    for (label, n, d) in cases {
        println!("{label:>16}  trials={n} dims={d}  →  {:?}", auto_pick(n, d));
    }
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
    fn small_low_dim_picks_grid() {
        // Exhaustive grid is feasible at low D and small N.
        assert_eq!(auto_pick(16, 2), HpoStrategy::Grid);
    }

    #[test]
    fn large_high_dim_picks_tpe() {
        // TPE shines with ≥ 20 trials × ≥ 4 dims.
        assert_eq!(auto_pick(100, 6), HpoStrategy::Tpe);
    }

    #[test]
    fn zero_trials_falls_back_to_random() {
        assert_eq!(auto_pick(0, 4), HpoStrategy::Random);
    }

    #[test]
    fn zero_dims_falls_back_to_random() {
        assert_eq!(auto_pick(50, 0), HpoStrategy::Random);
    }

    #[test]
    fn medium_3d_picks_random() {
        // 30 trials, 3 dims: too many for grid, too few dims for TPE → random.
        assert_eq!(auto_pick(30, 3), HpoStrategy::Random);
    }

    #[test]
    fn boundary_at_grid_limit() {
        assert_eq!(auto_pick(GRID_TRIAL_LIMIT, 2), HpoStrategy::Grid);
        assert_eq!(auto_pick(GRID_TRIAL_LIMIT + 1, 2), HpoStrategy::Random);
    }

    #[test]
    fn known_strategies_round_trip() {
        for s in ["grid", "random", "tpe"] {
            assert!(HpoStrategy::from_str_strict(s).is_some());
        }
        assert!(HpoStrategy::from_str_strict("bayesopt").is_none());
    }
}
