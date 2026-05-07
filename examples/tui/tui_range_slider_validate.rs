//! # TUI Range Slider Validate
//!
//! Validate a range slider's `[low, high]` selection given the
//! allowed `[min, max]` bounds and a step granularity. Returns
//! validated range or a categorical reason.
//!
//! Demonstrates the **TUI.154** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 `<input type="range" step>` semantics; jQuery UI
//!  slider min/max/step constraints.
//!
//! Run with: cargo run --example tui_range_slider_validate
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RangeVerdict {
    Ok { low: i32, high: i32 },
    OutOfBounds,
    InvertedRange,
    NotOnStep,
    InvalidConfig,
}

pub fn validate(low: i32, high: i32, min: i32, max: i32, step: u32) -> RangeVerdict {
    if min >= max || step == 0 {
        return RangeVerdict::InvalidConfig;
    }
    if low > high {
        return RangeVerdict::InvertedRange;
    }
    if low < min || high > max {
        return RangeVerdict::OutOfBounds;
    }
    let step = step as i32;
    if (low - min).rem_euclid(step) != 0 || (high - min).rem_euclid(step) != 0 {
        return RangeVerdict::NotOnStep;
    }
    RangeVerdict::Ok { low, high }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_range_slider_validate")?;

    println!("valid: {:?}", validate(10, 50, 0, 100, 5));
    println!("oob: {:?}", validate(-5, 50, 0, 100, 5));
    println!("inverted: {:?}", validate(50, 10, 0, 100, 5));
    println!("not on step: {:?}", validate(7, 50, 0, 100, 5));
    println!("invalid: {:?}", validate(0, 100, 100, 0, 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_range() {
        let v = validate(10, 50, 0, 100, 5);
        assert_eq!(v, RangeVerdict::Ok { low: 10, high: 50 });
    }

    #[test]
    fn invalid_min_ge_max() {
        assert_eq!(validate(0, 50, 100, 0, 5), RangeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_step() {
        assert_eq!(validate(10, 50, 0, 100, 0), RangeVerdict::InvalidConfig);
    }

    #[test]
    fn inverted_range_rejected() {
        assert_eq!(validate(50, 10, 0, 100, 5), RangeVerdict::InvertedRange);
    }

    #[test]
    fn low_below_min_oob() {
        assert_eq!(validate(-5, 50, 0, 100, 5), RangeVerdict::OutOfBounds);
    }

    #[test]
    fn high_above_max_oob() {
        assert_eq!(validate(0, 105, 0, 100, 5), RangeVerdict::OutOfBounds);
    }

    #[test]
    fn not_on_step_rejected() {
        assert_eq!(validate(7, 50, 0, 100, 5), RangeVerdict::NotOnStep);
    }

    #[test]
    fn boundary_at_min() {
        let v = validate(0, 50, 0, 100, 5);
        assert_eq!(v, RangeVerdict::Ok { low: 0, high: 50 });
    }

    #[test]
    fn boundary_at_max() {
        let v = validate(50, 100, 0, 100, 5);
        assert_eq!(v, RangeVerdict::Ok { low: 50, high: 100 });
    }

    #[test]
    fn equal_low_high_valid() {
        let v = validate(50, 50, 0, 100, 5);
        assert_eq!(v, RangeVerdict::Ok { low: 50, high: 50 });
    }

    #[test]
    fn deterministic() {
        let r1 = validate(10, 50, 0, 100, 5);
        let r2 = validate(10, 50, 0, 100, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn negative_min_handled() {
        let v = validate(-50, 0, -100, 100, 25);
        assert_eq!(v, RangeVerdict::Ok { low: -50, high: 0 });
    }

    #[test]
    fn step_one_accepts_any_int() {
        let v = validate(7, 13, 0, 100, 1);
        assert_eq!(v, RangeVerdict::Ok { low: 7, high: 13 });
    }
}
