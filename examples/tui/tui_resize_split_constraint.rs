//! # TUI Resize Split Constraint
//!
//! Distribute width between two split-pane children with min-size
//! constraints. Returns final widths or InsufficientWidth if both
//! mins exceed available space.
//!
//! Demonstrates the **TUI.105** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim split-pane resize semantics; CSS flex-basis with
//!  flex-shrink/grow.
//!
//! Run with: cargo run --example tui_resize_split_constraint
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok { left: u32, right: u32 },
    InsufficientWidth,
    InvalidConfig,
}

pub fn distribute(
    total_width: u32,
    left_min: u32,
    right_min: u32,
    left_ratio: f64,
) -> SplitVerdict {
    if total_width == 0 || !(0.0..=1.0).contains(&left_ratio) {
        return SplitVerdict::InvalidConfig;
    }
    if left_min + right_min > total_width {
        return SplitVerdict::InsufficientWidth;
    }
    let raw_left = (f64::from(total_width) * left_ratio) as u32;
    let mut left = raw_left.max(left_min);
    if total_width - left < right_min {
        left = total_width - right_min;
    }
    let right = total_width - left;
    SplitVerdict::Ok { left, right }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_resize_split_constraint")?;

    println!("balanced: {:?}", distribute(100, 20, 20, 0.5));
    println!("min-respected: {:?}", distribute(100, 80, 30, 0.5));
    println!("insufficient: {:?}", distribute(50, 30, 30, 0.5));
    println!("invalid: {:?}", distribute(0, 20, 20, 0.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn balanced_split() {
        let v = distribute(100, 20, 20, 0.5);
        if let SplitVerdict::Ok { left, right } = v {
            assert_eq!(left, 50);
            assert_eq!(right, 50);
        }
    }

    #[test]
    fn left_min_respected() {
        let v = distribute(100, 70, 20, 0.5);
        if let SplitVerdict::Ok { left, .. } = v {
            assert!(left >= 70);
        }
    }

    #[test]
    fn right_min_respected() {
        let v = distribute(100, 10, 70, 0.9);
        if let SplitVerdict::Ok { right, .. } = v {
            assert!(right >= 70);
        }
    }

    #[test]
    fn insufficient_width_returns_error() {
        assert_eq!(distribute(50, 30, 30, 0.5), SplitVerdict::InsufficientWidth);
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(distribute(0, 10, 10, 0.5), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_ratio_out_of_range() {
        assert_eq!(distribute(100, 10, 10, 1.5), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = distribute(100, 20, 20, 0.5);
        let r2 = distribute(100, 20, 20, 0.5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn left_plus_right_equals_total() {
        let v = distribute(100, 20, 20, 0.3);
        if let SplitVerdict::Ok { left, right } = v {
            assert_eq!(left + right, 100);
        }
    }

    #[test]
    fn ratio_zero_minimum_left() {
        let v = distribute(100, 10, 10, 0.0);
        if let SplitVerdict::Ok { left, .. } = v {
            assert_eq!(left, 10);
        }
    }

    #[test]
    fn ratio_one_maximum_left() {
        let v = distribute(100, 10, 10, 1.0);
        if let SplitVerdict::Ok { left, right } = v {
            assert_eq!(left, 90);
            assert_eq!(right, 10);
        }
    }

    #[test]
    fn ratio_seventy_thirty() {
        let v = distribute(100, 10, 10, 0.7);
        if let SplitVerdict::Ok { left, .. } = v {
            assert_eq!(left, 70);
        }
    }
}
