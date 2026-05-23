//! # TUI Dropdown Filter
//!
//! Filter dropdown items by query, return matches sorted by:
//!   1. exact match (highest)
//!   2. prefix match
//!   3. substring match
//!   4. case-insensitive substring
//!
//! Demonstrates the **TUI.12** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: fzf scoring algorithm + readline tab completion.
//!
//! Run with: cargo run --example tui_dropdown_filter
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok { matches: Vec<String> },
    EmptyItems,
}

pub fn filter(items: &[&str], query: &str) -> FilterVerdict {
    if items.is_empty() {
        return FilterVerdict::EmptyItems;
    }
    let q = query.trim();
    if q.is_empty() {
        return FilterVerdict::Ok {
            matches: items.iter().map(|s| (*s).to_string()).collect(),
        };
    }
    let q_lower = q.to_ascii_lowercase();
    let mut scored: Vec<(u32, &str)> = Vec::new();
    for item in items {
        let item_lower = item.to_ascii_lowercase();
        if *item == q {
            scored.push((4, *item));
        } else if item.starts_with(q) {
            scored.push((3, *item));
        } else if item.contains(q) {
            scored.push((2, *item));
        } else if item_lower.contains(&q_lower) {
            scored.push((1, *item));
        }
    }
    scored.sort_by_key(|b| std::cmp::Reverse(b.0));
    FilterVerdict::Ok {
        matches: scored.into_iter().map(|(_, s)| s.to_string()).collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_dropdown_filter")?;

    let items = ["apple", "apricot", "Apple Pie", "banana", "AppleSeed"];
    println!("query=apple: {:?}", filter(&items, "apple"));
    println!("query=Ap: {:?}", filter(&items, "Ap"));
    println!("query=banana: {:?}", filter(&items, "banana"));
    println!("query=zzz: {:?}", filter(&items, "zzz"));
    println!("empty query: {:?}", filter(&items, ""));
    println!("empty items: {:?}", filter(&[], "apple"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fruits() -> Vec<&'static str> {
        vec!["apple", "apricot", "Apple Pie", "banana", "AppleSeed"]
    }

    #[test]
    fn filterer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_first() {
        let v = filter(&fruits(), "apple");
        if let FilterVerdict::Ok { matches } = v {
            assert_eq!(matches[0], "apple");
        }
    }

    #[test]
    fn prefix_match_returned() {
        let v = filter(&fruits(), "ap");
        if let FilterVerdict::Ok { matches } = v {
            assert!(matches.iter().any(|s| s == "apple"));
            assert!(matches.iter().any(|s| s == "apricot"));
        }
    }

    #[test]
    fn case_insensitive_match() {
        let v = filter(&fruits(), "APPLE");
        if let FilterVerdict::Ok { matches } = v {
            // case-insensitive substring should pick up "apple" / "Apple Pie" / "AppleSeed".
            assert!(!matches.is_empty());
        }
    }

    #[test]
    fn no_matches_empty() {
        let v = filter(&fruits(), "zzz");
        if let FilterVerdict::Ok { matches } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn empty_query_returns_all() {
        let v = filter(&fruits(), "");
        if let FilterVerdict::Ok { matches } = v {
            assert_eq!(matches.len(), 5);
        }
    }

    #[test]
    fn whitespace_query_returns_all() {
        let v = filter(&fruits(), "   ");
        if let FilterVerdict::Ok { matches } = v {
            assert_eq!(matches.len(), 5);
        }
    }

    #[test]
    fn empty_items_rejected() {
        assert_eq!(filter(&[], "apple"), FilterVerdict::EmptyItems);
    }

    #[test]
    fn exact_outranks_prefix() {
        let items = ["apple", "apple_pie"];
        let v = filter(&items, "apple");
        if let FilterVerdict::Ok { matches } = v {
            assert_eq!(matches[0], "apple");
        }
    }

    #[test]
    fn unicode_query_works() {
        let items = ["café", "calé", "caves"];
        let v = filter(&items, "café");
        if let FilterVerdict::Ok { matches } = v {
            assert!(matches.iter().any(|s| s == "café"));
        }
    }

    #[test]
    fn deterministic() {
        let items = fruits();
        let a = filter(&items, "ap");
        let b = filter(&items, "ap");
        assert_eq!(a, b);
    }
}
