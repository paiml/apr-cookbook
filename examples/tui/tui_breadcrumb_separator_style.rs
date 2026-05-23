//! # TUI Breadcrumb Separator Style
//!
//! Render a breadcrumb path with a configurable separator (` / `,
//! ` › `, ` > `, ` :: `, etc). Returns rendered string and
//! per-segment positions.
//!
//! Demonstrates the **TUI.103** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Finder ` ▸ ` separator; Windows Explorer ` > `;
//!  HTML breadcrumb separator conventions.
//!
//! Run with: cargo run --example tui_breadcrumb_separator_style
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BreadcrumbVerdict {
    Ok {
        rendered: String,
        positions: Vec<u32>,
    },
    InvalidConfig,
}

pub fn render(segments: &[&str], separator: &str) -> BreadcrumbVerdict {
    if segments.is_empty() {
        return BreadcrumbVerdict::InvalidConfig;
    }
    let mut rendered = String::new();
    let mut positions: Vec<u32> = Vec::with_capacity(segments.len());
    let mut cursor: u32 = 0;
    for (i, seg) in segments.iter().enumerate() {
        positions.push(cursor);
        rendered.push_str(seg);
        cursor += seg.chars().count() as u32;
        if i + 1 < segments.len() {
            rendered.push_str(separator);
            cursor += separator.chars().count() as u32;
        }
    }
    BreadcrumbVerdict::Ok {
        rendered,
        positions,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_breadcrumb_separator_style")?;

    let segs = ["root", "src", "examples"];
    println!("slash: {:?}", render(&segs, " / "));
    println!("triangle: {:?}", render(&segs, " ▸ "));
    println!("double-colon: {:?}", render(&segs, " :: "));
    println!("invalid: {:?}", render(&[], " / "));
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
    fn slash_separator_correct() {
        let v = render(&["a", "b"], " / ");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "a / b");
        }
    }

    #[test]
    fn triangle_separator_correct() {
        let v = render(&["a", "b"], " ▸ ");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "a ▸ b");
        }
    }

    #[test]
    fn empty_separator_concatenates() {
        let v = render(&["a", "b"], "");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "ab");
        }
    }

    #[test]
    fn empty_segments_rejected() {
        assert_eq!(render(&[], " / "), BreadcrumbVerdict::InvalidConfig);
    }

    #[test]
    fn positions_count_matches_segments() {
        let v = render(&["a", "b", "c"], " / ");
        if let BreadcrumbVerdict::Ok { positions, .. } = v {
            assert_eq!(positions.len(), 3);
        }
    }

    #[test]
    fn first_position_zero() {
        let v = render(&["a", "b"], " / ");
        if let BreadcrumbVerdict::Ok { positions, .. } = v {
            assert_eq!(positions[0], 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["a"], " / ");
        let r2 = render(&["a"], " / ");
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_segment_no_separator() {
        let v = render(&["only"], " / ");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "only");
        }
    }

    #[test]
    fn unicode_separator_supported() {
        let v = render(&["a", "b"], " ▸ ");
        if let BreadcrumbVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('▸'));
        }
    }

    #[test]
    fn positions_advance_with_segments() {
        let v = render(&["aa", "bb"], " / ");
        if let BreadcrumbVerdict::Ok { positions, .. } = v {
            assert_eq!(positions[0], 0);
            assert_eq!(positions[1], 5); // "aa" + " / " = 5 chars
        }
    }

    #[test]
    fn many_segments_handled() {
        let segs: Vec<&str> = vec!["x"; 10];
        let v = render(&segs, " / ");
        if let BreadcrumbVerdict::Ok { positions, .. } = v {
            assert_eq!(positions.len(), 10);
        }
    }
}
