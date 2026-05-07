//! # TUI Search Input Box
//!
//! Render a search input box with placeholder text, cursor, and
//! current query. Returns the rendered string and effective cursor
//! position.
//!
//! Demonstrates the **TUI.176** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Spotlight / fzf search-bar UX; HTML5 input
//!  placeholder semantics.
//!
//! Run with: cargo run --example tui_search_input_box
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok { rendered: String, cursor_col: u32 },
    InvalidConfig,
}

pub fn render(query: &str, placeholder: &str, width: u32, cursor_pos: u32) -> SearchVerdict {
    if width < 5 || cursor_pos > query.chars().count() as u32 {
        return SearchVerdict::InvalidConfig;
    }
    let display_text = if query.is_empty() {
        placeholder.to_string()
    } else {
        query.to_string()
    };
    let truncated: String = display_text.chars().take(width as usize - 2).collect();
    let cursor_col = if query.is_empty() {
        1
    } else {
        // Account for the leading "🔍" prefix (1 col) plus offset.
        cursor_pos + 1
    };
    let rendered = format!("🔍{truncated:<w$}", w = (width - 2) as usize);
    SearchVerdict::Ok {
        rendered,
        cursor_col,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_search_input_box")?;

    println!("placeholder: {:?}", render("", "Search...", 30, 0));
    println!("typed: {:?}", render("hello", "Search...", 30, 5));
    println!("invalid: {:?}", render("x", "p", 3, 0));
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
    fn invalid_too_narrow() {
        assert_eq!(render("x", "p", 3, 0), SearchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_cursor_oob() {
        assert_eq!(render("hi", "p", 30, 100), SearchVerdict::InvalidConfig);
    }

    #[test]
    fn placeholder_when_empty() {
        let v = render("", "Search...", 30, 0);
        if let SearchVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("Search"));
        }
    }

    #[test]
    fn query_displayed() {
        let v = render("hello", "p", 30, 5);
        if let SearchVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("hello"));
        }
    }

    #[test]
    fn search_glyph_present() {
        let v = render("x", "p", 30, 1);
        if let SearchVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('🔍'));
        }
    }

    #[test]
    fn cursor_col_correct_for_typed() {
        let v = render("hello", "p", 30, 5);
        if let SearchVerdict::Ok { cursor_col, .. } = v {
            assert_eq!(cursor_col, 6);
        }
    }

    #[test]
    fn cursor_col_one_for_empty() {
        let v = render("", "p", 30, 0);
        if let SearchVerdict::Ok { cursor_col, .. } = v {
            assert_eq!(cursor_col, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("x", "p", 30, 1);
        let r2 = render("x", "p", 30, 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn long_query_truncated() {
        let v = render("a very long search query string", "p", 15, 0);
        if let SearchVerdict::Ok { rendered, .. } = v {
            // 15 width - 2 for glyph = 13 chars max
            assert!(rendered.contains("a very long s"));
        }
    }

    #[test]
    fn unicode_query_supported() {
        let v = render("café", "p", 30, 4);
        if let SearchVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn min_width_accepted() {
        let v = render("a", "p", 5, 1);
        assert!(matches!(v, SearchVerdict::Ok { .. }));
    }

    #[test]
    fn cursor_at_zero_with_query_pos_one() {
        let v = render("abc", "p", 30, 0);
        if let SearchVerdict::Ok { cursor_col, .. } = v {
            assert_eq!(cursor_col, 1);
        }
    }
}
