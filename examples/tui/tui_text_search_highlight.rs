//! # TUI Text Search Highlight
//!
//! Find all positions of a query in text. Returns (start, end) char
//! positions for highlighting. Case-insensitive variant available.
//!
//! Demonstrates the **TUI.19** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: less/grep highlight conventions.
//!
//! Run with: cargo run --example tui_text_search_highlight
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HighlightVerdict {
    Ok { matches: Vec<(usize, usize)> },
    EmptyQuery,
    EmptyText,
}

pub fn find_all(text: &str, query: &str, case_sensitive: bool) -> HighlightVerdict {
    if text.is_empty() {
        return HighlightVerdict::EmptyText;
    }
    if query.is_empty() {
        return HighlightVerdict::EmptyQuery;
    }
    let haystack: String = if case_sensitive {
        text.to_string()
    } else {
        text.to_ascii_lowercase()
    };
    let needle: String = if case_sensitive {
        query.to_string()
    } else {
        query.to_ascii_lowercase()
    };
    let h_chars: Vec<char> = haystack.chars().collect();
    let n_chars: Vec<char> = needle.chars().collect();
    let n = n_chars.len();
    let mut matches = Vec::new();
    if n > h_chars.len() {
        return HighlightVerdict::Ok { matches };
    }
    for i in 0..=(h_chars.len() - n) {
        if h_chars[i..i + n] == n_chars[..] {
            matches.push((i, i + n));
        }
    }
    HighlightVerdict::Ok { matches }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_text_search_highlight")?;

    println!(
        "case sensitive: {:?}",
        find_all("Hello world hello", "hello", true)
    );
    println!(
        "case insensitive: {:?}",
        find_all("Hello world hello", "hello", false)
    );
    println!("not found: {:?}", find_all("hello", "xyz", true));
    println!("empty query: {:?}", find_all("hello", "", true));
    println!("empty text: {:?}", find_all("", "x", true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn case_sensitive_finds_one() {
        let v = find_all("Hello world hello", "hello", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert_eq!(matches.len(), 1);
        }
    }

    #[test]
    fn case_insensitive_finds_two() {
        let v = find_all("Hello world hello", "hello", false);
        if let HighlightVerdict::Ok { matches } = v {
            assert_eq!(matches.len(), 2);
        }
    }

    #[test]
    fn no_match_empty_list() {
        let v = find_all("hello", "xyz", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn empty_query_rejected() {
        assert_eq!(find_all("hello", "", true), HighlightVerdict::EmptyQuery);
    }

    #[test]
    fn empty_text_rejected() {
        assert_eq!(find_all("", "x", true), HighlightVerdict::EmptyText);
    }

    #[test]
    fn match_positions_correct() {
        let v = find_all("aaa", "a", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert_eq!(matches, vec![(0, 1), (1, 2), (2, 3)]);
        }
    }

    #[test]
    fn query_longer_than_text_empty() {
        let v = find_all("abc", "abcdef", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn full_match_works() {
        let v = find_all("hello", "hello", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert_eq!(matches, vec![(0, 5)]);
        }
    }

    #[test]
    fn unicode_search() {
        let v = find_all("café au lait", "café", true);
        if let HighlightVerdict::Ok { matches } = v {
            assert_eq!(matches, vec![(0, 4)]);
        }
    }

    #[test]
    fn deterministic() {
        let a = find_all("hello world", "world", true);
        let b = find_all("hello world", "world", true);
        assert_eq!(a, b);
    }
}
