//! # Contracts-Macros Inverse Postcondition
//!
//! Some postconditions can be inverted: given an output and tolerance,
//! infer the valid input range. Useful for fuzz-test corpus generation
//! and Lean lemma verification.
//!
//! Demonstrates the **CMM.07** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Refinement-typed inverse contracts (Liquid Haskell).
//!
//! Run with: cargo run --example contracts_macros_inverse_postcond
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum InverseVerdict {
    Ok { input_min: f64, input_max: f64 },
    Unbounded,
    NotInvertible,
    InvalidConfig,
}

pub fn invert_linear(output: f64, slope: f64, intercept: f64, tolerance: f64) -> InverseVerdict {
    if !output.is_finite()
        || !slope.is_finite()
        || !intercept.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.0
    {
        return InverseVerdict::InvalidConfig;
    }
    if slope == 0.0 {
        return InverseVerdict::NotInvertible;
    }
    let lo = (output - tolerance - intercept) / slope;
    let hi = (output + tolerance - intercept) / slope;
    let (input_min, input_max) = if slope > 0.0 { (lo, hi) } else { (hi, lo) };
    if !input_min.is_finite() || !input_max.is_finite() {
        return InverseVerdict::Unbounded;
    }
    InverseVerdict::Ok {
        input_min,
        input_max,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_inverse_postcond")?;

    println!("y=2x+1: {:?}", invert_linear(11.0, 2.0, 1.0, 0.5));
    println!("negative slope: {:?}", invert_linear(11.0, -2.0, 1.0, 0.5));
    println!("not invertible: {:?}", invert_linear(11.0, 0.0, 1.0, 0.5));
    println!("invalid: {:?}", invert_linear(11.0, 2.0, 1.0, -1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_inverts_correctly() {
        let v = invert_linear(11.0, 2.0, 1.0, 0.5);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            // y = 2x+1, y∈[10.5, 11.5] → x ∈ [4.75, 5.25].
            assert!((input_min - 4.75).abs() < 1e-9);
            assert!((input_max - 5.25).abs() < 1e-9);
        }
    }

    #[test]
    fn negative_slope_orders_bounds() {
        let v = invert_linear(11.0, -2.0, 1.0, 0.5);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            assert!(input_min < input_max);
        }
    }

    #[test]
    fn zero_slope_not_invertible() {
        assert_eq!(
            invert_linear(11.0, 0.0, 1.0, 0.5),
            InverseVerdict::NotInvertible
        );
    }

    #[test]
    fn negative_tolerance_invalid() {
        assert_eq!(
            invert_linear(11.0, 2.0, 1.0, -1.0),
            InverseVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            invert_linear(f64::NAN, 2.0, 1.0, 0.5),
            InverseVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_tolerance_returns_point() {
        let v = invert_linear(11.0, 2.0, 1.0, 0.0);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            assert!((input_min - input_max).abs() < 1e-9);
        }
    }

    #[test]
    fn input_min_at_correct_endpoint() {
        let v = invert_linear(0.0, 1.0, 0.0, 1.0);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            assert!((input_min - -1.0).abs() < 1e-9);
            assert!((input_max - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn very_steep_slope_narrow_range() {
        let v = invert_linear(0.0, 1000.0, 0.0, 1.0);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            // Range is 2/1000 = 0.002.
            assert!((input_max - input_min - 0.002).abs() < 1e-9);
        }
    }

    #[test]
    fn shallow_slope_wide_range() {
        let v = invert_linear(0.0, 0.001, 0.0, 1.0);
        if let InverseVerdict::Ok {
            input_min,
            input_max,
        } = v
        {
            // Range is 2 / 0.001 = 2000.
            assert!((input_max - input_min - 2000.0).abs() < 1e-3);
        }
    }

    #[test]
    fn deterministic() {
        let a = invert_linear(11.0, 2.0, 1.0, 0.5);
        let b = invert_linear(11.0, 2.0, 1.0, 0.5);
        assert_eq!(a, b);
    }
}
