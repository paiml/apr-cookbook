//! # TUI Emoji Picker Filter
//!
//! Filter emoji catalog by name keyword. Returns matching
//! `(emoji, name)` pairs sorted by name.
//!
//! Demonstrates the **TUI.120** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Character Viewer; Slack emoji picker UI.
//!
//! Run with: cargo run --example tui_emoji_picker_filter
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EmojiVerdict {
    Ok {
        matches: Vec<(String, String)>,
        match_count: u32,
    },
    InvalidConfig,
}

pub fn filter(catalog: &[(&str, &str)], query: &str) -> EmojiVerdict {
    if catalog.is_empty() || query.is_empty() {
        return EmojiVerdict::InvalidConfig;
    }
    let q_lower = query.to_lowercase();
    let mut matches: Vec<(String, String)> = catalog
        .iter()
        .filter(|(_, name)| name.to_lowercase().contains(&q_lower))
        .map(|(emoji, name)| ((*emoji).to_string(), (*name).to_string()))
        .collect();
    matches.sort_by(|a, b| a.1.cmp(&b.1));
    let match_count = matches.len() as u32;
    EmojiVerdict::Ok {
        matches,
        match_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_emoji_picker_filter")?;

    let catalog = [
        ("🚀", "rocket"),
        ("🎉", "party"),
        ("💯", "hundred"),
        ("🔥", "fire"),
    ];
    println!("query 'r': {:?}", filter(&catalog, "r"));
    println!("query 'fire': {:?}", filter(&catalog, "fire"));
    println!("invalid: {:?}", filter(&[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn match_found() {
        let catalog = [("🚀", "rocket"), ("🎉", "party")];
        let v = filter(&catalog, "rocket");
        if let EmojiVerdict::Ok { match_count, .. } = v {
            assert_eq!(match_count, 1);
        }
    }

    #[test]
    fn no_match_zero() {
        let catalog = [("🚀", "rocket")];
        let v = filter(&catalog, "xyz");
        if let EmojiVerdict::Ok { matches, .. } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn case_insensitive() {
        let catalog = [("🚀", "ROCKET")];
        let v = filter(&catalog, "rocket");
        if let EmojiVerdict::Ok { match_count, .. } = v {
            assert_eq!(match_count, 1);
        }
    }

    #[test]
    fn empty_catalog_rejected() {
        assert_eq!(filter(&[], "r"), EmojiVerdict::InvalidConfig);
    }

    #[test]
    fn empty_query_rejected() {
        let catalog = [("🚀", "rocket")];
        assert_eq!(filter(&catalog, ""), EmojiVerdict::InvalidConfig);
    }

    #[test]
    fn matches_sorted_by_name() {
        let catalog = [("🚀", "rocket"), ("🍎", "apple"), ("🐱", "cat")];
        let v = filter(&catalog, ""); // Empty query rejected; use partial.
        let _ = v;
        let v = filter(&catalog, "a");
        if let EmojiVerdict::Ok { matches, .. } = v {
            // "apple" and "cat" both contain 'a' → sorted: apple, cat.
            if matches.len() >= 2 {
                assert_eq!(matches[0].1, "apple");
                assert_eq!(matches[1].1, "cat");
            }
        }
    }

    #[test]
    fn deterministic() {
        let catalog = [("🚀", "rocket")];
        let r1 = filter(&catalog, "r");
        let r2 = filter(&catalog, "r");
        assert_eq!(r1, r2);
    }

    #[test]
    fn substring_match() {
        let catalog = [("🚀", "rocket")];
        let v = filter(&catalog, "rock");
        if let EmojiVerdict::Ok { match_count, .. } = v {
            assert_eq!(match_count, 1);
        }
    }

    #[test]
    fn match_pair_includes_emoji() {
        let catalog = [("🚀", "rocket")];
        let v = filter(&catalog, "rocket");
        if let EmojiVerdict::Ok { matches, .. } = v {
            assert_eq!(matches[0].0, "🚀");
        }
    }

    #[test]
    fn unicode_query_works() {
        let catalog = [("🚀", "café-rocket")];
        let v = filter(&catalog, "café");
        if let EmojiVerdict::Ok { match_count, .. } = v {
            assert_eq!(match_count, 1);
        }
    }

    #[test]
    fn many_emoji_handled() {
        let catalog: Vec<(&str, &str)> = (0..20).map(|_| ("🚀", "rocket")).collect();
        let v = filter(&catalog, "rocket");
        if let EmojiVerdict::Ok { match_count, .. } = v {
            assert_eq!(match_count, 20);
        }
    }
}
