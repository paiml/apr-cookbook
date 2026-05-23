//! # TUI Chart Threshold Overlay
//!
//! Compute the row index where a horizontal threshold line should be
//! drawn over a chart given (chart_height, value_min, value_max,
//! threshold). Returns row + whether threshold is in-bounds.
//!
//! Demonstrates the **TUI.114** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: matplotlib axhline; Grafana threshold band conventions.
//!
//! Run with: cargo run --example tui_chart_threshold_overlay
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OverlayVerdict {
    Ok { row_index: u32, in_bounds: bool },
    InvalidConfig,
}

pub fn compute(
    chart_height: u32,
    value_min: f64,
    value_max: f64,
    threshold: f64,
) -> OverlayVerdict {
    if chart_height < 2
        || !value_min.is_finite()
        || !value_max.is_finite()
        || !threshold.is_finite()
        || value_max <= value_min
    {
        return OverlayVerdict::InvalidConfig;
    }
    let in_bounds = (value_min..=value_max).contains(&threshold);
    let normalized = (threshold - value_min) / (value_max - value_min);
    let row_from_bottom = (normalized * f64::from(chart_height - 1)) as u32;
    let row_index = (chart_height - 1).saturating_sub(row_from_bottom);
    OverlayVerdict::Ok {
        row_index,
        in_bounds,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_chart_threshold_overlay")?;

    println!("middle: {:?}", compute(20, 0.0, 100.0, 50.0));
    println!("at top: {:?}", compute(20, 0.0, 100.0, 100.0));
    println!("out of range: {:?}", compute(20, 0.0, 100.0, 150.0));
    println!("invalid: {:?}", compute(0, 0.0, 100.0, 50.0));
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
    fn middle_threshold_middle_row() {
        let v = compute(11, 0.0, 100.0, 50.0);
        if let OverlayVerdict::Ok { row_index, .. } = v {
            // 11 rows → middle is index 5.
            assert_eq!(row_index, 5);
        }
    }

    #[test]
    fn top_threshold_top_row() {
        let v = compute(20, 0.0, 100.0, 100.0);
        if let OverlayVerdict::Ok { row_index, .. } = v {
            assert_eq!(row_index, 0);
        }
    }

    #[test]
    fn bottom_threshold_bottom_row() {
        let v = compute(20, 0.0, 100.0, 0.0);
        if let OverlayVerdict::Ok { row_index, .. } = v {
            assert_eq!(row_index, 19);
        }
    }

    #[test]
    fn out_of_range_high_marked() {
        let v = compute(20, 0.0, 100.0, 150.0);
        if let OverlayVerdict::Ok { in_bounds, .. } = v {
            assert!(!in_bounds);
        }
    }

    #[test]
    fn out_of_range_low_marked() {
        let v = compute(20, 0.0, 100.0, -10.0);
        if let OverlayVerdict::Ok { in_bounds, .. } = v {
            assert!(!in_bounds);
        }
    }

    #[test]
    fn invalid_zero_height() {
        assert_eq!(compute(0, 0.0, 100.0, 50.0), OverlayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_one_height() {
        assert_eq!(compute(1, 0.0, 100.0, 50.0), OverlayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_max_le_min() {
        assert_eq!(
            compute(20, 100.0, 100.0, 50.0),
            OverlayVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_nan_threshold() {
        assert_eq!(
            compute(20, 0.0, 100.0, f64::NAN),
            OverlayVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let r1 = compute(20, 0.0, 100.0, 50.0);
        let r2 = compute(20, 0.0, 100.0, 50.0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn row_index_in_chart_bounds() {
        let v = compute(20, 0.0, 100.0, 50.0);
        if let OverlayVerdict::Ok { row_index, .. } = v {
            assert!(row_index < 20);
        }
    }

    #[test]
    fn negative_range_works() {
        let v = compute(11, -100.0, 100.0, 0.0);
        if let OverlayVerdict::Ok { row_index, .. } = v {
            assert_eq!(row_index, 5);
        }
    }
}
