//! # Distillation Per-Class Temperature
//!
//! Different classes benefit from different temperatures in soft-label
//! distillation. Picker:
//!   minority classes → higher T (smooth more)
//!   common classes → lower T (preserve detail)
//!
//! Demonstrates the **DIST.29** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015) on temperature scaling.
//!
//! Run with: cargo run --example distill_per_class_temperature
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TempVerdict {
    Ok { per_class_temps: Vec<f64> },
    EmptyClasses,
    AllZero,
}

pub fn pick(class_counts: &[u64]) -> TempVerdict {
    if class_counts.is_empty() {
        return TempVerdict::EmptyClasses;
    }
    let max_count = class_counts.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        return TempVerdict::AllZero;
    }
    let per_class_temps: Vec<f64> = class_counts
        .iter()
        .map(|&c| {
            // Ratio inversely proportional to size: minority gets higher T.
            let ratio = c as f64 / max_count as f64;
            // Linear from T_min=2.0 (most common) to T_max=8.0 (rarest).
            // T = 8.0 - 6.0 × ratio.
            8.0 - 6.0 * ratio
        })
        .collect();
    TempVerdict::Ok { per_class_temps }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_per_class_temperature")?;

    println!("balanced: {:?}", pick(&[1000, 1000, 1000]));
    println!("imbalanced: {:?}", pick(&[10000, 100, 50]));
    println!("empty: {:?}", pick(&[]));
    println!("all zero: {:?}", pick(&[0, 0, 0]));
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
    fn balanced_classes_same_temp() {
        let v = pick(&[1000, 1000, 1000]);
        if let TempVerdict::Ok { per_class_temps } = v {
            assert!((per_class_temps[0] - per_class_temps[1]).abs() < 1e-9);
            assert!((per_class_temps[1] - per_class_temps[2]).abs() < 1e-9);
        }
    }

    #[test]
    fn rarest_gets_highest_temp() {
        let v = pick(&[10000, 100, 50]);
        if let TempVerdict::Ok { per_class_temps } = v {
            // 50 (rarest) > 100 > 10000 (most common).
            assert!(per_class_temps[2] > per_class_temps[1]);
            assert!(per_class_temps[1] > per_class_temps[0]);
        }
    }

    #[test]
    fn most_common_gets_t_2() {
        let v = pick(&[1000, 100, 50]);
        if let TempVerdict::Ok { per_class_temps } = v {
            // Most common ratio = 1.0 → T = 8 - 6*1 = 2.0.
            assert!((per_class_temps[0] - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(pick(&[]), TempVerdict::EmptyClasses);
    }

    #[test]
    fn all_zero_rejected() {
        assert_eq!(pick(&[0, 0, 0]), TempVerdict::AllZero);
    }

    #[test]
    fn temps_in_range_2_to_8() {
        let v = pick(&[10000, 100, 50, 1]);
        if let TempVerdict::Ok { per_class_temps } = v {
            for t in &per_class_temps {
                assert!(*t >= 2.0);
                assert!(*t <= 8.0);
            }
        }
    }

    #[test]
    fn class_count_preserved() {
        let v = pick(&[100, 200, 300, 400, 500]);
        if let TempVerdict::Ok { per_class_temps } = v {
            assert_eq!(per_class_temps.len(), 5);
        }
    }

    #[test]
    fn rarest_at_t_8_when_minimal() {
        let v = pick(&[1_000_000, 1]);
        if let TempVerdict::Ok { per_class_temps } = v {
            // Ratio 1/1M ≈ 0 → T ≈ 8.0.
            assert!((per_class_temps[1] - 8.0).abs() < 1e-3);
        }
    }

    #[test]
    fn single_class_returns_t_2() {
        let v = pick(&[100]);
        if let TempVerdict::Ok { per_class_temps } = v {
            assert_eq!(per_class_temps.len(), 1);
            assert!((per_class_temps[0] - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(&[10000, 100, 50]);
        let b = pick(&[10000, 100, 50]);
        assert_eq!(a, b);
    }
}
