//! # TUI Table Pagination Indicator
//!
//! Render `Page N of M` style indicator with prev/next button states
//! (enabled/disabled at boundaries).
//!
//! Demonstrates the **TUI.104** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: REST API pagination conventions; Material Design
//!  Pagination component.
//!
//! Run with: cargo run --example tui_table_pagination_indicator
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PageVerdict {
    Ok {
        rendered: String,
        prev_enabled: bool,
        next_enabled: bool,
    },
    InvalidConfig,
}

pub fn render(current_page: u32, total_pages: u32) -> PageVerdict {
    if total_pages == 0 || current_page == 0 || current_page > total_pages {
        return PageVerdict::InvalidConfig;
    }
    let prev_arrow = if current_page > 1 { "<" } else { " " };
    let next_arrow = if current_page < total_pages { ">" } else { " " };
    let rendered = format!("[{prev_arrow}] Page {current_page} of {total_pages} [{next_arrow}]");
    PageVerdict::Ok {
        rendered,
        prev_enabled: current_page > 1,
        next_enabled: current_page < total_pages,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_table_pagination_indicator")?;

    println!("middle: {:?}", render(3, 12));
    println!("first: {:?}", render(1, 12));
    println!("last: {:?}", render(12, 12));
    println!("invalid: {:?}", render(0, 12));
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
    fn middle_both_enabled() {
        let v = render(3, 12);
        if let PageVerdict::Ok {
            prev_enabled,
            next_enabled,
            ..
        } = v
        {
            assert!(prev_enabled);
            assert!(next_enabled);
        }
    }

    #[test]
    fn first_page_prev_disabled() {
        let v = render(1, 12);
        if let PageVerdict::Ok { prev_enabled, .. } = v {
            assert!(!prev_enabled);
        }
    }

    #[test]
    fn last_page_next_disabled() {
        let v = render(12, 12);
        if let PageVerdict::Ok { next_enabled, .. } = v {
            assert!(!next_enabled);
        }
    }

    #[test]
    fn rendered_contains_page_text() {
        let v = render(3, 12);
        if let PageVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("Page 3 of 12"));
        }
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(render(0, 0), PageVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_current() {
        assert_eq!(render(0, 12), PageVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_current_gt_total() {
        assert_eq!(render(15, 12), PageVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = render(3, 12);
        let r2 = render(3, 12);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_page_neither_enabled() {
        let v = render(1, 1);
        if let PageVerdict::Ok {
            prev_enabled,
            next_enabled,
            ..
        } = v
        {
            assert!(!prev_enabled);
            assert!(!next_enabled);
        }
    }

    #[test]
    fn brackets_in_rendered() {
        let v = render(3, 12);
        if let PageVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('['));
            assert!(rendered.contains(']'));
        }
    }

    #[test]
    fn arrow_chars_present_when_enabled() {
        let v = render(5, 12);
        if let PageVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('<'));
            assert!(rendered.contains('>'));
        }
    }

    #[test]
    fn arrow_chars_absent_when_disabled() {
        let v = render(1, 1);
        if let PageVerdict::Ok { rendered, .. } = v {
            assert!(!rendered.contains('<'));
            assert!(!rendered.contains('>'));
        }
    }
}
