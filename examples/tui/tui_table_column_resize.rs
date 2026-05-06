//! # TUI Table Column Resize
//!
//! Distribute available `total_width` columns among N table columns
//! using min/max constraints. Returns final widths or InvalidConfig
//! if even all-min totals exceed available width.
//!
//! Demonstrates the **TUI.96** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ratatui Constraint::Min/Max layout solver; CSS Grid
//!  fr-unit allocation.
//!
//! Run with: cargo run --example tui_table_column_resize
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ResizeVerdict {
    Ok { widths: Vec<u32> },
    InsufficientWidth,
    InvalidConfig,
}

pub fn distribute(
    constraints: &[(u32, u32)], // (min, max)
    total_width: u32,
) -> ResizeVerdict {
    if constraints.is_empty() || total_width == 0 {
        return ResizeVerdict::InvalidConfig;
    }
    if constraints.iter().any(|(lo, hi)| lo > hi) {
        return ResizeVerdict::InvalidConfig;
    }
    let total_min: u32 = constraints.iter().map(|(lo, _)| lo).sum();
    if total_min > total_width {
        return ResizeVerdict::InsufficientWidth;
    }
    let mut widths: Vec<u32> = constraints.iter().map(|(lo, _)| *lo).collect();
    let mut remaining = total_width - total_min;
    while remaining > 0 {
        let mut grew_any = false;
        for (i, (_, hi)) in constraints.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            if widths[i] < *hi {
                widths[i] += 1;
                remaining -= 1;
                grew_any = true;
            }
        }
        if !grew_any {
            break;
        }
    }
    ResizeVerdict::Ok { widths }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_column_resize")?;

    let cons = [(5u32, 20), (10u32, 30), (5u32, 15)];
    println!("60 cols: {:?}", distribute(&cons, 60));
    println!("12 cols: {:?}", distribute(&cons, 12));
    println!("invalid: {:?}", distribute(&[], 60));
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
    fn min_widths_satisfied() {
        let cons = [(5u32, 20), (10u32, 30)];
        let v = distribute(&cons, 40);
        if let ResizeVerdict::Ok { widths } = v {
            assert!(widths[0] >= 5);
            assert!(widths[1] >= 10);
        }
    }

    #[test]
    fn max_widths_respected() {
        let cons = [(5u32, 20), (10u32, 30)];
        let v = distribute(&cons, 100);
        if let ResizeVerdict::Ok { widths } = v {
            assert!(widths[0] <= 20);
            assert!(widths[1] <= 30);
        }
    }

    #[test]
    fn insufficient_width_returns_error() {
        let cons = [(50u32, 100)];
        assert_eq!(distribute(&cons, 10), ResizeVerdict::InsufficientWidth);
    }

    #[test]
    fn invalid_min_gt_max_rejected() {
        let cons = [(50u32, 10)];
        assert_eq!(distribute(&cons, 100), ResizeVerdict::InvalidConfig);
    }

    #[test]
    fn empty_constraints_rejected() {
        assert_eq!(distribute(&[], 100), ResizeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_width_rejected() {
        let cons = [(5u32, 20)];
        assert_eq!(distribute(&cons, 0), ResizeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let cons = [(5u32, 20), (10u32, 30)];
        let r1 = distribute(&cons, 60);
        let r2 = distribute(&cons, 60);
        assert_eq!(r1, r2);
    }

    #[test]
    fn widths_count_matches_constraints() {
        let cons = [(5u32, 20), (10u32, 30), (5u32, 15)];
        let v = distribute(&cons, 60);
        if let ResizeVerdict::Ok { widths } = v {
            assert_eq!(widths.len(), 3);
        }
    }

    #[test]
    fn at_min_total_no_extra() {
        let cons = [(5u32, 20), (10u32, 30)];
        let v = distribute(&cons, 15);
        if let ResizeVerdict::Ok { widths } = v {
            assert_eq!(widths[0], 5);
            assert_eq!(widths[1], 10);
        }
    }

    #[test]
    fn extra_width_distributed_round_robin() {
        let cons = [(0u32, 100), (0u32, 100)];
        let v = distribute(&cons, 6);
        if let ResizeVerdict::Ok { widths } = v {
            assert_eq!(widths[0] + widths[1], 6);
            // Round-robin → equal split.
            assert_eq!(widths[0], widths[1]);
        }
    }

    #[test]
    fn one_column_takes_all() {
        let cons = [(0u32, 1000)];
        let v = distribute(&cons, 50);
        if let ResizeVerdict::Ok { widths } = v {
            assert_eq!(widths[0], 50);
        }
    }
}
