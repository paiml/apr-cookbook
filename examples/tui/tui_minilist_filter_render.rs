//! # TUI Minilist Filter Render
//!
//! Filter a list by a query string; non-matching entries are dimmed
//! (rendered with `[ ]` prefix) while matches show `[*]` prefix.
//! Returns rendered lines and matched count.
//!
//! Demonstrates the **TUI.112** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ratatui filtered list pattern; VS Code Quick Open
//!  match-highlight UX.
//!
//! Run with: cargo run --example tui_minilist_filter_render
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MinilistVerdict {
    Ok {
        lines: Vec<String>,
        matched_count: u32,
    },
    InvalidConfig,
}

pub fn render(items: &[&str], query: &str) -> MinilistVerdict {
    if items.is_empty() {
        return MinilistVerdict::InvalidConfig;
    }
    let q_lower = query.to_lowercase();
    let mut lines: Vec<String> = Vec::with_capacity(items.len());
    let mut matched_count = 0u32;
    for item in items {
        let prefix = if !query.is_empty() && item.to_lowercase().contains(&q_lower) {
            matched_count += 1;
            "[*]"
        } else {
            "[ ]"
        };
        lines.push(format!("{prefix} {item}"));
    }
    MinilistVerdict::Ok {
        lines,
        matched_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_minilist_filter_render")?;

    let items = ["alpha", "beta", "gamma"];
    println!("query 'a': {:?}", render(&items, "a"));
    println!("no query: {:?}", render(&items, ""));
    println!("invalid: {:?}", render(&[], ""));
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
    fn match_marked_with_star() {
        let v = render(&["alpha"], "alpha");
        if let MinilistVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("[*]"));
        }
    }

    #[test]
    fn non_match_marked_with_blank() {
        let v = render(&["alpha"], "xyz");
        if let MinilistVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("[ ]"));
        }
    }

    #[test]
    fn empty_query_no_matches() {
        let v = render(&["alpha"], "");
        if let MinilistVerdict::Ok { matched_count, .. } = v {
            assert_eq!(matched_count, 0);
        }
    }

    #[test]
    fn case_insensitive_match() {
        let v = render(&["Alpha"], "ALPHA");
        if let MinilistVerdict::Ok { matched_count, .. } = v {
            assert_eq!(matched_count, 1);
        }
    }

    #[test]
    fn empty_items_rejected() {
        assert_eq!(render(&[], "x"), MinilistVerdict::InvalidConfig);
    }

    #[test]
    fn matched_count_correct() {
        let v = render(&["alpha", "beta", "alphabet"], "alpha");
        if let MinilistVerdict::Ok { matched_count, .. } = v {
            assert_eq!(matched_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["a"], "a");
        let r2 = render(&["a"], "a");
        assert_eq!(r1, r2);
    }

    #[test]
    fn lines_count_matches_items() {
        let v = render(&["a", "b", "c"], "x");
        if let MinilistVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn item_text_in_line() {
        let v = render(&["alpha"], "x");
        if let MinilistVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("alpha"));
        }
    }

    #[test]
    fn unicode_query_supported() {
        let v = render(&["café"], "café");
        if let MinilistVerdict::Ok { matched_count, .. } = v {
            assert_eq!(matched_count, 1);
        }
    }

    #[test]
    fn no_match_zero_count() {
        let v = render(&["a", "b"], "zzz");
        if let MinilistVerdict::Ok { matched_count, .. } = v {
            assert_eq!(matched_count, 0);
        }
    }
}
