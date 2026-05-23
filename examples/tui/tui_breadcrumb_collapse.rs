//! # TUI Breadcrumb Collapse
//!
//! Collapse middle segments of a breadcrumb path with `…` when the
//! rendered width exceeds `max_width`. Always keeps first + last
//! segment visible.
//!
//! Demonstrates the **TUI.58** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Finder + VS Code breadcrumb truncation UX.
//!
//! Run with: cargo run --example tui_breadcrumb_collapse
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CollapseVerdict {
    Ok { rendered: String, collapsed: bool },
    InvalidConfig,
}

pub fn render(segments: &[&str], separator: &str, max_width: u32) -> CollapseVerdict {
    if segments.is_empty() || max_width == 0 {
        return CollapseVerdict::InvalidConfig;
    }
    let full = segments.join(separator);
    let full_w = full.chars().count() as u32;
    if full_w <= max_width {
        return CollapseVerdict::Ok {
            rendered: full,
            collapsed: false,
        };
    }
    if segments.len() < 3 {
        // Can't collapse middle of < 3 segments — return as-is.
        return CollapseVerdict::Ok {
            rendered: full,
            collapsed: false,
        };
    }
    let first = segments[0];
    let last = segments[segments.len() - 1];
    let rendered = format!("{first}{separator}…{separator}{last}");
    CollapseVerdict::Ok {
        rendered,
        collapsed: true,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_breadcrumb_collapse")?;

    let segments = ["root", "src", "examples", "tui"];
    println!("fits: {:?}", render(&segments, " / ", 100));
    println!("collapsed: {:?}", render(&segments, " / ", 15));
    println!("invalid: {:?}", render(&[], " / ", 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fits_no_collapse() {
        let v = render(&["root", "src"], " / ", 100);
        if let CollapseVerdict::Ok {
            rendered,
            collapsed,
        } = v
        {
            assert!(!collapsed);
            assert_eq!(rendered, "root / src");
        }
    }

    #[test]
    fn long_path_collapses() {
        let v = render(&["a", "b", "c", "d", "e"], " / ", 8);
        if let CollapseVerdict::Ok {
            rendered,
            collapsed,
        } = v
        {
            assert!(collapsed);
            assert!(rendered.contains('…'));
            assert!(rendered.starts_with('a'));
            assert!(rendered.ends_with('e'));
        }
    }

    #[test]
    fn empty_segments_rejected() {
        assert_eq!(render(&[], " / ", 10), CollapseVerdict::InvalidConfig);
    }

    #[test]
    fn zero_width_rejected() {
        assert_eq!(render(&["a"], " / ", 0), CollapseVerdict::InvalidConfig);
    }

    #[test]
    fn single_segment_no_collapse() {
        let v = render(&["only"], " / ", 1);
        if let CollapseVerdict::Ok {
            rendered,
            collapsed,
        } = v
        {
            assert!(!collapsed);
            assert_eq!(rendered, "only");
        }
    }

    #[test]
    fn two_segments_no_collapse_even_long() {
        let v = render(&["root", "src"], " / ", 5);
        if let CollapseVerdict::Ok { collapsed, .. } = v {
            assert!(!collapsed);
        }
    }

    #[test]
    fn first_and_last_preserved() {
        let v = render(&["alpha", "x", "y", "z", "omega"], " / ", 10);
        if let CollapseVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("alpha"));
            assert!(rendered.contains("omega"));
        }
    }

    #[test]
    fn unicode_segments() {
        let v = render(&["café", "x", "y", "résumé"], " / ", 10);
        if let CollapseVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
            assert!(rendered.contains("résumé"));
        }
    }

    #[test]
    fn deterministic() {
        let segs = ["a", "b", "c", "d"];
        let a = render(&segs, " / ", 5);
        let b = render(&segs, " / ", 5);
        assert_eq!(a, b);
    }

    #[test]
    fn other_separator_works() {
        let v = render(&["a", "b", "c", "d"], " > ", 5);
        if let CollapseVerdict::Ok {
            rendered,
            collapsed,
        } = v
        {
            assert!(collapsed);
            assert!(rendered.contains(" > "));
        }
    }

    #[test]
    fn collapsed_string_does_not_contain_middle_segments() {
        let v = render(&["A", "MIDDLE", "B"], " / ", 3);
        if let CollapseVerdict::Ok { rendered, .. } = v {
            assert!(!rendered.contains("MIDDLE"));
        }
    }
}
