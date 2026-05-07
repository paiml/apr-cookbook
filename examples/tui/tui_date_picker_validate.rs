//! # TUI Date Picker Validate
//!
//! Validate a YYYY-MM-DD date input string and check it falls within
//! the allowed range. Returns parsed components or a categorical
//! `Invalid` reason.
//!
//! Demonstrates the **TUI.152** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ISO 8601 §5.2.1 calendar-date format; HTML5 `<input
//!  type="date">` validation rules.
//!
//! Run with: cargo run --example tui_date_picker_validate
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DateVerdict {
    Ok { year: u32, month: u32, day: u32 },
    BadFormat,
    OutOfRange,
    InvalidConfig,
}

pub fn validate(input: &str, min_year: u32, max_year: u32) -> DateVerdict {
    if min_year > max_year || max_year > 9999 {
        return DateVerdict::InvalidConfig;
    }
    if input.len() != 10 || !input.is_ascii() {
        return DateVerdict::BadFormat;
    }
    let bytes = input.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return DateVerdict::BadFormat;
    }
    let parse = |s: &str| -> Option<u32> { s.parse::<u32>().ok() };
    let Some(year) = parse(&input[0..4]) else {
        return DateVerdict::BadFormat;
    };
    let Some(month) = parse(&input[5..7]) else {
        return DateVerdict::BadFormat;
    };
    let Some(day) = parse(&input[8..10]) else {
        return DateVerdict::BadFormat;
    };
    if !(1..=12).contains(&month) {
        return DateVerdict::OutOfRange;
    }
    let max_day = days_in_month(year, month);
    if !(1..=max_day).contains(&day) {
        return DateVerdict::OutOfRange;
    }
    if year < min_year || year > max_year {
        return DateVerdict::OutOfRange;
    }
    DateVerdict::Ok { year, month, day }
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_date_picker_validate")?;

    println!("valid: {:?}", validate("2026-05-07", 2000, 2030));
    println!("oob month: {:?}", validate("2026-13-07", 2000, 2030));
    println!("bad format: {:?}", validate("2026/05/07", 2000, 2030));
    println!("invalid: {:?}", validate("2026-05-07", 2030, 2000));
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
    fn valid_date() {
        let v = validate("2026-05-07", 2000, 2030);
        assert_eq!(
            v,
            DateVerdict::Ok {
                year: 2026,
                month: 5,
                day: 7,
            }
        );
    }

    #[test]
    fn invalid_min_above_max() {
        assert_eq!(
            validate("2026-05-07", 2030, 2000),
            DateVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_year_too_high() {
        assert_eq!(validate("2026-05-07", 0, 99999), DateVerdict::InvalidConfig);
    }

    #[test]
    fn bad_length_format() {
        assert_eq!(validate("2026-05-7", 2000, 2030), DateVerdict::BadFormat);
    }

    #[test]
    fn bad_separator_format() {
        assert_eq!(validate("2026/05/07", 2000, 2030), DateVerdict::BadFormat);
    }

    #[test]
    fn month_zero_oob() {
        assert_eq!(validate("2026-00-15", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn month_thirteen_oob() {
        assert_eq!(validate("2026-13-15", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn day_zero_oob() {
        assert_eq!(validate("2026-05-00", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn day_32_in_jan_oob() {
        assert_eq!(validate("2026-01-32", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn feb_29_leap_year_valid() {
        let v = validate("2024-02-29", 2000, 2030);
        assert_eq!(
            v,
            DateVerdict::Ok {
                year: 2024,
                month: 2,
                day: 29,
            }
        );
    }

    #[test]
    fn feb_29_non_leap_oob() {
        assert_eq!(validate("2025-02-29", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn year_below_min_oob() {
        assert_eq!(validate("1999-05-07", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn year_above_max_oob() {
        assert_eq!(validate("2031-05-07", 2000, 2030), DateVerdict::OutOfRange);
    }

    #[test]
    fn deterministic() {
        let r1 = validate("2026-05-07", 2000, 2030);
        let r2 = validate("2026-05-07", 2000, 2030);
        assert_eq!(r1, r2);
    }

    #[test]
    fn century_year_not_leap() {
        // 1900 is divisible by 100 but not 400 → not a leap year.
        assert_eq!(validate("1900-02-29", 1800, 2000), DateVerdict::OutOfRange);
    }
}
