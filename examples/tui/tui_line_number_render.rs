//! # TUI Line Number Gutter Render
//!
//! Render right-aligned line numbers for a buffer of N lines, padded
//! to the width of the largest line number. Returns the gutter
//! strings and the column width used.
//!
//! Demonstrates the **TUI.133** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:set number`/`:set rnu` rendering; emacs
//!  `linum-mode` and `display-line-numbers-mode`.
//!
//! Run with: cargo run --example tui_line_number_render
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GutterVerdict {
    Ok {
        gutter_lines: Vec<String>,
        col_width: u32,
    },
    InvalidConfig,
}

pub fn render(line_count: u32, start_line: u32) -> GutterVerdict {
    if line_count == 0 || start_line == 0 {
        return GutterVerdict::InvalidConfig;
    }
    let last = start_line + line_count - 1;
    let width = digit_count(last);
    let mut gutter: Vec<String> = Vec::with_capacity(line_count as usize);
    for n in start_line..=last {
        gutter.push(format!("{n:>width$}", width = width as usize));
    }
    GutterVerdict::Ok {
        gutter_lines: gutter,
        col_width: width,
    }
}

fn digit_count(n: u32) -> u32 {
    let mut x = n;
    let mut d = 1u32;
    while x >= 10 {
        x /= 10;
        d += 1;
    }
    d
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_line_number_render")?;

    println!("3 lines from 1: {:?}", render(3, 1));
    println!("100 lines from 99: {:?}", render(3, 99));
    println!("invalid: {:?}", render(0, 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn line_count_matches() {
        let v = render(5, 1);
        if let GutterVerdict::Ok { gutter_lines, .. } = v {
            assert_eq!(gutter_lines.len(), 5);
        }
    }

    #[test]
    fn col_width_one_for_single_digit() {
        let v = render(9, 1);
        if let GutterVerdict::Ok { col_width, .. } = v {
            assert_eq!(col_width, 1);
        }
    }

    #[test]
    fn col_width_two_for_two_digit() {
        let v = render(50, 1);
        if let GutterVerdict::Ok { col_width, .. } = v {
            assert_eq!(col_width, 2);
        }
    }

    #[test]
    fn col_width_three_for_three_digit() {
        let v = render(150, 1);
        if let GutterVerdict::Ok { col_width, .. } = v {
            assert_eq!(col_width, 3);
        }
    }

    #[test]
    fn invalid_zero_count() {
        assert_eq!(render(0, 1), GutterVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_start() {
        assert_eq!(render(5, 0), GutterVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = render(3, 1);
        let r2 = render(3, 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn padding_aligned() {
        let v = render(3, 99);
        if let GutterVerdict::Ok { gutter_lines, .. } = v {
            // 99,100,101 → all padded to 3 chars
            for l in &gutter_lines {
                assert_eq!(l.len(), 3);
            }
        }
    }

    #[test]
    fn first_line_correct() {
        let v = render(3, 1);
        if let GutterVerdict::Ok { gutter_lines, .. } = v {
            assert_eq!(gutter_lines[0].trim(), "1");
        }
    }

    #[test]
    fn last_line_correct() {
        let v = render(3, 1);
        if let GutterVerdict::Ok { gutter_lines, .. } = v {
            assert_eq!(gutter_lines[2].trim(), "3");
        }
    }

    #[test]
    fn high_start_handled() {
        // render(2, 9999) → lines 9999, 10000 → max digit count = 5.
        let v = render(2, 9999);
        if let GutterVerdict::Ok { col_width, .. } = v {
            assert_eq!(col_width, 5);
        }
    }

    #[test]
    fn many_lines_handled() {
        let v = render(1000, 1);
        if let GutterVerdict::Ok { gutter_lines, .. } = v {
            assert_eq!(gutter_lines.len(), 1000);
        }
    }
}
