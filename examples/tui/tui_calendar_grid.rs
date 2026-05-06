//! # TUI Calendar Month Grid
//!
//! Compute a 7-column month grid: which weekday is the 1st, day-
//! cells per row (0 = blank), and total weeks rendered. Sunday-start
//! by default.
//!
//! Demonstrates the **TUI.57** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cal(1) Unix calendar utility.
//!
//! Run with: cargo run --example tui_calendar_grid
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CalendarVerdict {
    Ok {
        first_weekday: u32,
        cells: Vec<u32>,
        weeks: u32,
    },
    InvalidConfig,
}

pub fn build(days_in_month: u32, first_weekday_sunday0: u32) -> CalendarVerdict {
    if days_in_month == 0 || days_in_month > 31 || first_weekday_sunday0 > 6 {
        return CalendarVerdict::InvalidConfig;
    }
    let mut cells: Vec<u32> = vec![0; first_weekday_sunday0 as usize];
    cells.extend(1..=days_in_month);
    // Pad trailing blanks to multiple of 7.
    let pad = (7 - cells.len() % 7) % 7;
    cells.resize(cells.len() + pad, 0);
    let weeks = (cells.len() / 7) as u32;
    CalendarVerdict::Ok {
        first_weekday: first_weekday_sunday0,
        cells,
        weeks,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_calendar_grid")?;

    println!("Jan starts Wed: {:?}", build(31, 3));
    println!("Feb (28d, Tues start): {:?}", build(28, 2));
    println!("invalid: {:?}", build(0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn cells_multiple_of_seven() {
        let v = build(31, 3);
        if let CalendarVerdict::Ok { cells, .. } = v {
            assert_eq!(cells.len() % 7, 0);
        }
    }

    #[test]
    fn first_day_starts_at_correct_col() {
        let v = build(31, 3);
        if let CalendarVerdict::Ok { cells, .. } = v {
            assert_eq!(cells[0], 0);
            assert_eq!(cells[3], 1);
        }
    }

    #[test]
    fn day_count_correct() {
        let v = build(28, 0);
        if let CalendarVerdict::Ok { cells, .. } = v {
            let day_count = cells.iter().filter(|d| **d > 0).count();
            assert_eq!(day_count, 28);
        }
    }

    #[test]
    fn invalid_zero_days() {
        assert_eq!(build(0, 0), CalendarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_many_days() {
        assert_eq!(build(32, 0), CalendarVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_high_weekday() {
        assert_eq!(build(28, 7), CalendarVerdict::InvalidConfig);
    }

    #[test]
    fn weeks_matches_grid_height() {
        let v = build(31, 6);
        if let CalendarVerdict::Ok { cells, weeks, .. } = v {
            assert_eq!(weeks * 7, cells.len() as u32);
        }
    }

    #[test]
    fn shortest_month_4_weeks() {
        // Feb 28 starting Sunday → 4 full weeks.
        let v = build(28, 0);
        if let CalendarVerdict::Ok { weeks, .. } = v {
            assert_eq!(weeks, 4);
        }
    }

    #[test]
    fn longest_month_6_weeks() {
        // 31 days starting Saturday → 6 weeks.
        let v = build(31, 6);
        if let CalendarVerdict::Ok { weeks, .. } = v {
            assert_eq!(weeks, 6);
        }
    }

    #[test]
    fn last_day_present() {
        let v = build(31, 3);
        if let CalendarVerdict::Ok { cells, .. } = v {
            assert!(cells.contains(&31));
        }
    }

    #[test]
    fn deterministic() {
        let a = build(28, 2);
        let b = build(28, 2);
        assert_eq!(a, b);
    }
}
