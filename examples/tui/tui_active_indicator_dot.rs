//! # TUI Active Indicator Dot
//!
//! Render a list of items with the active item marked by a leading
//! dot indicator (●). Returns the rendered lines.
//!
//! Demonstrates the **TUI.168** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS sidebar active-item glyph; ARIA `aria-current`
//!  rendering convention.
//!
//! Run with: cargo run --example tui_active_indicator_dot
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IndicatorVerdict {
    Ok { lines: Vec<String>, active_idx: u32 },
    InvalidConfig,
}

pub fn render(items: &[&str], active_idx: u32) -> IndicatorVerdict {
    if items.is_empty() || (active_idx as usize) >= items.len() {
        return IndicatorVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let prefix = if (i as u32) == active_idx {
            "● "
        } else {
            "  "
        };
        lines.push(format!("{prefix}{item}"));
    }
    IndicatorVerdict::Ok { lines, active_idx }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_active_indicator_dot")?;

    println!("items: {:?}", render(&["a", "b", "c"], 1));
    println!("invalid: {:?}", render(&[], 0));
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
    fn empty_items_rejected() {
        assert_eq!(render(&[], 0), IndicatorVerdict::InvalidConfig);
    }

    #[test]
    fn active_idx_oob_rejected() {
        assert_eq!(render(&["a"], 5), IndicatorVerdict::InvalidConfig);
    }

    #[test]
    fn active_item_has_dot() {
        let v = render(&["a", "b"], 0);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[0].starts_with("● "));
        }
    }

    #[test]
    fn inactive_item_no_dot() {
        let v = render(&["a", "b"], 0);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[1].starts_with("  "));
            assert!(!lines[1].contains('●'));
        }
    }

    #[test]
    fn line_count_matches_items() {
        let v = render(&["a", "b", "c"], 1);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn active_idx_returned() {
        let v = render(&["a", "b", "c"], 2);
        if let IndicatorVerdict::Ok { active_idx, .. } = v {
            assert_eq!(active_idx, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["a"], 0);
        let r2 = render(&["a"], 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn item_text_present() {
        let v = render(&["hello"], 0);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("hello"));
        }
    }

    #[test]
    fn middle_active_correct() {
        let v = render(&["a", "b", "c"], 1);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[0].starts_with("  "));
            assert!(lines[1].starts_with("● "));
            assert!(lines[2].starts_with("  "));
        }
    }

    #[test]
    fn last_active_correct() {
        let v = render(&["a", "b", "c"], 2);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[2].starts_with("● "));
        }
    }

    #[test]
    fn unicode_item_supported() {
        let v = render(&["café", "résumé"], 0);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("café"));
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<&str> = (0..30).map(|_| "item").collect();
        let v = render(&items, 15);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 30);
            assert!(lines[15].starts_with("● "));
        }
    }

    #[test]
    fn single_item_active() {
        let v = render(&["only"], 0);
        if let IndicatorVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 1);
            assert!(lines[0].starts_with("● "));
        }
    }
}
