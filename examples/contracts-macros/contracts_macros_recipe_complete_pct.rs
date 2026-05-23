//! # Contracts-Macros Recipe Complete Pct
//!
//! Compute completeness percentage of a recipe based on weighted
//! field presence. Returns the percentage and the weighted gap to
//! 100%.
//!
//! Demonstrates the **CMM.201** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ISO/IEC 25010 completeness sub-characteristic; SLSA
//!  attestation completeness scoring.
//!
//! Run with: cargo run --example contracts_macros_recipe_complete_pct
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompleteVerdict {
    Ok { completeness_pct: u32, gap_pct: u32 },
    InvalidConfig,
}

/// Items: (field_name, weight, present).
pub fn compute(fields: &[(&str, u32, bool)]) -> CompleteVerdict {
    if fields.is_empty() {
        return CompleteVerdict::InvalidConfig;
    }
    let total_weight: u32 = fields.iter().map(|(_, w, _)| *w).sum();
    if total_weight == 0 {
        return CompleteVerdict::InvalidConfig;
    }
    let present_weight: u32 = fields
        .iter()
        .filter(|(_, _, present)| *present)
        .map(|(_, w, _)| *w)
        .sum();
    let pct = (present_weight as u64 * 100 / total_weight as u64) as u32;
    CompleteVerdict::Ok {
        completeness_pct: pct,
        gap_pct: 100 - pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_complete_pct")?;

    let fields = [
        ("title", 30, true),
        ("body", 50, true),
        ("citation", 20, false),
    ];
    println!("compute: {:?}", compute(&fields));
    println!("invalid: {:?}", compute(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_present_100_pct() {
        let v = compute(&[("a", 50, true), ("b", 50, true)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 100);
        }
    }

    #[test]
    fn none_present_0_pct() {
        let v = compute(&[("a", 50, false), ("b", 50, false)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 0);
        }
    }

    #[test]
    fn weighted_partial_correct() {
        // 30+50 / 100 = 80%
        let v = compute(&[("a", 30, true), ("b", 50, true), ("c", 20, false)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 80);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compute(&[]), CompleteVerdict::InvalidConfig);
    }

    #[test]
    fn zero_total_weight_rejected() {
        assert_eq!(compute(&[("a", 0, true)]), CompleteVerdict::InvalidConfig);
    }

    #[test]
    fn gap_complements_pct() {
        let v = compute(&[("a", 30, true), ("b", 70, false)]);
        if let CompleteVerdict::Ok {
            completeness_pct,
            gap_pct,
        } = v
        {
            assert_eq!(completeness_pct + gap_pct, 100);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(&[("a", 50, true)]);
        let r2 = compute(&[("a", 50, true)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_field_handled() {
        let v = compute(&[("a", 100, true)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 100);
        }
    }

    #[test]
    fn many_fields_handled() {
        let fields: Vec<(&str, u32, bool)> = (0..30).map(|_| ("f", 1, true)).collect();
        let v = compute(&fields);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 100);
        }
    }

    #[test]
    fn unicode_field_supported() {
        let v = compute(&[("café", 100, true)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 100);
        }
    }

    #[test]
    fn pct_in_zero_to_100() {
        let v = compute(&[("a", 50, true), ("b", 50, false)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert!(completeness_pct <= 100);
        }
    }

    #[test]
    fn high_weights_handled() {
        let v = compute(&[("a", 1_000_000, true)]);
        if let CompleteVerdict::Ok {
            completeness_pct, ..
        } = v
        {
            assert_eq!(completeness_pct, 100);
        }
    }
}
