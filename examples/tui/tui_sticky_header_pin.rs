//! # TUI Sticky Header Pin
//!
//! Determine which (if any) section header should appear pinned at
//! the top of the viewport given the current scroll position. Headers
//! are sorted by line; the active one is the highest header at or
//! above the scroll line.
//!
//! Demonstrates the **TUI.145** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML `position: sticky`; macOS Finder column-view sticky
//!  group headers; iOS UITableView pinned section headers.
//!
//! Run with: cargo run --example tui_sticky_header_pin
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StickyVerdict {
    Ok {
        pinned_header: Option<String>,
        active_section_idx: u32,
    },
    InvalidConfig,
}

pub fn pin(headers: &[(u32, &str)], scroll_line: u32) -> StickyVerdict {
    if headers.is_empty() {
        return StickyVerdict::InvalidConfig;
    }
    // Headers must be in ascending line order.
    for w in headers.windows(2) {
        if w[0].0 >= w[1].0 {
            return StickyVerdict::InvalidConfig;
        }
    }
    let mut active_idx: Option<usize> = None;
    for (i, (line, _)) in headers.iter().enumerate() {
        if *line <= scroll_line {
            active_idx = Some(i);
        } else {
            break;
        }
    }
    match active_idx {
        Some(i) => StickyVerdict::Ok {
            pinned_header: Some(headers[i].1.to_string()),
            active_section_idx: i as u32,
        },
        None => StickyVerdict::Ok {
            pinned_header: None,
            active_section_idx: 0,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_sticky_header_pin")?;

    let headers = [(0u32, "Intro"), (10, "Setup"), (50, "Usage")];
    println!("scroll 5: {:?}", pin(&headers, 5));
    println!("scroll 60: {:?}", pin(&headers, 60));
    println!("invalid: {:?}", pin(&[], 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_headers_rejected() {
        assert_eq!(pin(&[], 10), StickyVerdict::InvalidConfig);
    }

    #[test]
    fn unsorted_headers_rejected() {
        assert_eq!(pin(&[(10, "b"), (5, "a")], 0), StickyVerdict::InvalidConfig);
    }

    #[test]
    fn duplicate_lines_rejected() {
        assert_eq!(pin(&[(5, "a"), (5, "b")], 0), StickyVerdict::InvalidConfig);
    }

    #[test]
    fn first_header_pinned_at_start() {
        let v = pin(&[(0, "Intro"), (10, "Setup")], 5);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("Intro".to_string()));
        }
    }

    #[test]
    fn no_pin_before_first_header() {
        let v = pin(&[(10, "Setup")], 5);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, None);
        }
    }

    #[test]
    fn boundary_at_header_line_pins() {
        let v = pin(&[(10, "Setup")], 10);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("Setup".to_string()));
        }
    }

    #[test]
    fn pins_highest_above_scroll() {
        let v = pin(&[(0, "A"), (10, "B"), (50, "C")], 30);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("B".to_string()));
        }
    }

    #[test]
    fn active_section_idx_correct() {
        let v = pin(&[(0, "A"), (10, "B"), (50, "C")], 60);
        if let StickyVerdict::Ok {
            active_section_idx, ..
        } = v
        {
            assert_eq!(active_section_idx, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = pin(&[(0, "A")], 5);
        let r2 = pin(&[(0, "A")], 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_headers_handled() {
        let headers: Vec<(u32, &str)> = (0..30).map(|i| (i * 10, "h")).collect();
        let v = pin(&headers, 200);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("h".to_string()));
        }
    }

    #[test]
    fn unicode_header_supported() {
        let v = pin(&[(0, "café")], 5);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("café".to_string()));
        }
    }

    #[test]
    fn high_scroll_pins_last() {
        let v = pin(&[(0, "A"), (10, "B")], 1_000_000);
        if let StickyVerdict::Ok { pinned_header, .. } = v {
            assert_eq!(pinned_header, Some("B".to_string()));
        }
    }
}
