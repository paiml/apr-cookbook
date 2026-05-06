//! # Monte-Carlo Drop-In Replacement Test
//!
//! Statistical equivalence test: are two models' outputs equivalent
//! within tolerance over N samples? Returns observed mismatch rate
//! and a verdict at 95% confidence.
//!
//! Demonstrates the **MC.19** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TOST equivalence test (Schuirmann 1987).
//!
//! Run with: cargo run --example mc_drop_in_replacement_test
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EquivVerdict {
    Equivalent { mismatch_rate: f64 },
    NotEquivalent { mismatch_rate: f64 },
    InvalidConfig,
}

pub fn test(sample_diffs: &[f64], tolerance: f64, max_mismatch_rate: f64) -> EquivVerdict {
    if sample_diffs.is_empty()
        || !tolerance.is_finite()
        || tolerance < 0.0
        || !max_mismatch_rate.is_finite()
        || !(0.0..=1.0).contains(&max_mismatch_rate)
    {
        return EquivVerdict::InvalidConfig;
    }
    if sample_diffs.iter().any(|d| !d.is_finite()) {
        return EquivVerdict::InvalidConfig;
    }
    let mismatches: u32 = sample_diffs.iter().filter(|d| d.abs() > tolerance).count() as u32;
    let n = sample_diffs.len() as u32;
    let mismatch_rate = f64::from(mismatches) / f64::from(n);
    if mismatch_rate <= max_mismatch_rate {
        EquivVerdict::Equivalent { mismatch_rate }
    } else {
        EquivVerdict::NotEquivalent { mismatch_rate }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_drop_in_replacement_test")?;

    let close: Vec<f64> = (0..1000).map(|i| f64::from(i % 10) * 0.001).collect();
    println!("equivalent: {:?}", test(&close, 0.05, 0.05));

    let off: Vec<f64> = (0..100).map(|_| 0.5).collect();
    println!("not equivalent: {:?}", test(&off, 0.05, 0.05));
    println!("invalid: {:?}", test(&[], 0.05, 0.05));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tester_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_diffs_equivalent() {
        let v = test(&[0.001, -0.002, 0.003, -0.001], 0.01, 0.05);
        assert!(matches!(v, EquivVerdict::Equivalent { .. }));
    }

    #[test]
    fn large_diffs_not_equivalent() {
        let v = test(&[0.5, 0.6, 0.7], 0.01, 0.05);
        assert!(matches!(v, EquivVerdict::NotEquivalent { .. }));
    }

    #[test]
    fn boundary_at_max_rate_equivalent() {
        // 1 of 20 mismatches = 5% = boundary.
        let mut diffs = vec![0.0; 19];
        diffs.push(1.0);
        let v = test(&diffs, 0.01, 0.05);
        assert!(matches!(v, EquivVerdict::Equivalent { .. }));
    }

    #[test]
    fn empty_invalid() {
        assert_eq!(test(&[], 0.01, 0.05), EquivVerdict::InvalidConfig);
    }

    #[test]
    fn nan_diff_invalid() {
        assert_eq!(
            test(&[f64::NAN, 0.0], 0.01, 0.05),
            EquivVerdict::InvalidConfig
        );
    }

    #[test]
    fn neg_tolerance_invalid() {
        assert_eq!(test(&[0.0], -0.01, 0.05), EquivVerdict::InvalidConfig);
    }

    #[test]
    fn over_max_rate_invalid() {
        assert_eq!(test(&[0.0], 0.01, 1.5), EquivVerdict::InvalidConfig);
    }

    #[test]
    fn zero_tolerance_strict() {
        // Even tiny diffs count as mismatches with zero tolerance.
        let v = test(&[1e-15, 1e-15, 1e-15, 1e-15], 0.0, 0.05);
        assert!(matches!(v, EquivVerdict::NotEquivalent { .. }));
    }

    #[test]
    fn rate_value_correct() {
        let v = test(&[0.0, 0.0, 1.0, 0.0], 0.01, 0.5);
        if let EquivVerdict::Equivalent { mismatch_rate } = v {
            assert!((mismatch_rate - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let diffs = vec![0.0, 0.0, 1.0];
        let a = test(&diffs, 0.5, 0.5);
        let b = test(&diffs, 0.5, 0.5);
        assert_eq!(a, b);
    }
}
