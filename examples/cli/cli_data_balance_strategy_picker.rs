//! # apr data balance — Strategy Picker (oversample/undersample/smote)
//!
//! `apr data balance --strategy <S>` accepts {oversample, undersample,
//! smote}. Picker rules: if class imbalance ratio < 2x → none needed;
//! if minority class < 100 samples → SMOTE; otherwise oversample. This
//! recipe builds the auto-picker.
//!
//! Demonstrates the **DATA-BALANCE.4** recipe for PMAT-106 (apr data balance coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DATA-BALANCE-001 + Chawla 2002 (SMOTE)
//!
//! Run with: cargo run --example cli_data_balance_strategy_picker
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceStrategy {
    None,
    Oversample,
    Undersample,
    Smote,
}

impl BalanceStrategy {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "none" => Some(BalanceStrategy::None),
            "oversample" => Some(BalanceStrategy::Oversample),
            "undersample" => Some(BalanceStrategy::Undersample),
            "smote" => Some(BalanceStrategy::Smote),
            _ => None,
        }
    }
}

const IMBALANCE_THRESHOLD: f64 = 2.0;
const SMOTE_MINORITY_FLOOR: u64 = 100;

pub fn auto_pick(min_class_count: u64, max_class_count: u64) -> BalanceStrategy {
    if min_class_count == 0 {
        // Pathological — no samples in some class; recommend none, caller error.
        return BalanceStrategy::None;
    }
    let ratio = max_class_count as f64 / min_class_count as f64;
    if ratio < IMBALANCE_THRESHOLD {
        return BalanceStrategy::None;
    }
    if min_class_count < SMOTE_MINORITY_FLOOR {
        BalanceStrategy::Smote
    } else {
        BalanceStrategy::Oversample
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_data_balance_strategy_picker")?;

    let cases = [
        ("balanced", 1000u64, 1100u64),
        ("mild imbalance", 100, 500),
        ("smote case", 30, 1000),
        ("tiny minority", 5, 10000),
        ("zero minority", 0, 100),
    ];
    for (label, min, max) in cases {
        println!("{label:>20}  ({min} / {max})  →  {:?}", auto_pick(min, max));
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
    fn balanced_picks_none() {
        // Below 2x imbalance → no rebalancing needed.
        assert_eq!(auto_pick(1000, 1500), BalanceStrategy::None);
    }

    #[test]
    fn imbalanced_above_floor_picks_oversample() {
        // 5x imbalance + 200-sample minority → oversample.
        assert_eq!(auto_pick(200, 1000), BalanceStrategy::Oversample);
    }

    #[test]
    fn imbalanced_below_floor_picks_smote() {
        // 5x imbalance + 30-sample minority → SMOTE (oversample would just
        // duplicate the same 30 samples).
        assert_eq!(auto_pick(30, 1000), BalanceStrategy::Smote);
    }

    #[test]
    fn boundary_just_below_2x_picks_none() {
        // < 2x imbalance (strict) → no rebalancing. 501/1000 ≈ 1.996x.
        assert_eq!(auto_pick(501, 1000), BalanceStrategy::None);
    }

    #[test]
    fn boundary_at_100_minority_picks_oversample() {
        // < SMOTE_MINORITY_FLOOR (100) → SMOTE.
        // == 100 → Oversample.
        assert_eq!(auto_pick(99, 1000), BalanceStrategy::Smote);
        assert_eq!(auto_pick(100, 1000), BalanceStrategy::Oversample);
    }

    #[test]
    fn zero_minority_returns_none() {
        // Pathological — defer to operator.
        assert_eq!(auto_pick(0, 100), BalanceStrategy::None);
    }

    #[test]
    fn known_strategies_round_trip() {
        for s in ["none", "oversample", "undersample", "smote"] {
            assert!(BalanceStrategy::from_str_strict(s).is_some());
        }
        assert!(BalanceStrategy::from_str_strict("typo").is_none());
    }
}
