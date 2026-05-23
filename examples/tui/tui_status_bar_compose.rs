//! # TUI Status-Bar Composer
//!
//! Pack left, center, right segments into a fixed-width status bar.
//! If contents overflow, return Truncated. Otherwise pad with spaces
//! to fill the bar.
//!
//! Demonstrates the **TUI.10** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim/tmux statusline composition.
//!
//! Run with: cargo run --example tui_status_bar_compose
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StatusVerdict {
    Ok { bar: String },
    Truncated { overflow_chars: usize },
    InvalidWidth,
}

pub fn compose(left: &str, center: &str, right: &str, width: usize) -> StatusVerdict {
    if width == 0 {
        return StatusVerdict::InvalidWidth;
    }
    let l = left.chars().count();
    let c = center.chars().count();
    let r = right.chars().count();
    let total = l + c + r;
    if total > width {
        return StatusVerdict::Truncated {
            overflow_chars: total - width,
        };
    }
    let center_start = (width - c) / 2;
    let mut bar: Vec<char> = vec![' '; width];
    for (i, ch) in left.chars().enumerate() {
        bar[i] = ch;
    }
    for (i, ch) in center.chars().enumerate() {
        bar[center_start + i] = ch;
    }
    let right_start = width - r;
    for (i, ch) in right.chars().enumerate() {
        bar[right_start + i] = ch;
    }
    StatusVerdict::Ok {
        bar: bar.iter().collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_status_bar_compose")?;

    println!("typical: {:?}", compose("[INSERT]", "main.rs", "1:1 ", 40));
    println!(
        "narrow truncates: {:?}",
        compose("very long left", "center", "very long right", 20)
    );
    println!("all empty: {:?}", compose("", "", "", 10));
    println!("invalid: {:?}", compose("a", "b", "c", 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_fits_width() {
        let v = compose("[L]", "C", "[R]", 20);
        if let StatusVerdict::Ok { bar } = v {
            assert_eq!(bar.chars().count(), 20);
            assert!(bar.starts_with("[L]"));
            assert!(bar.ends_with("[R]"));
        }
    }

    #[test]
    fn overflow_returns_truncated() {
        let v = compose("aaaa", "bbbb", "cccc", 5);
        assert!(matches!(v, StatusVerdict::Truncated { .. }));
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(compose("a", "b", "c", 0), StatusVerdict::InvalidWidth);
    }

    #[test]
    fn empty_segments_padded() {
        let v = compose("", "", "", 10);
        if let StatusVerdict::Ok { bar } = v {
            assert_eq!(bar, "          ");
        }
    }

    #[test]
    fn center_in_middle() {
        let v = compose("", "X", "", 5);
        if let StatusVerdict::Ok { bar } = v {
            // Middle of 5 = index 2.
            assert_eq!(bar.chars().nth(2), Some('X'));
        }
    }

    #[test]
    fn left_at_start() {
        let v = compose("LL", "", "", 10);
        if let StatusVerdict::Ok { bar } = v {
            assert!(bar.starts_with("LL"));
        }
    }

    #[test]
    fn right_at_end() {
        let v = compose("", "", "RR", 10);
        if let StatusVerdict::Ok { bar } = v {
            assert!(bar.ends_with("RR"));
        }
    }

    #[test]
    fn unicode_segments() {
        let v = compose("←", "★", "→", 10);
        if let StatusVerdict::Ok { bar } = v {
            assert!(bar.contains('←'));
            assert!(bar.contains('★'));
            assert!(bar.contains('→'));
        }
    }

    #[test]
    fn exact_fit_no_truncation() {
        let v = compose("aa", "bb", "cc", 6);
        assert!(matches!(v, StatusVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let a = compose("L", "C", "R", 10);
        let b = compose("L", "C", "R", 10);
        assert_eq!(a, b);
    }
}
