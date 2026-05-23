//! # TUI Text Search Match Count
//!
//! Count occurrences of `needle` in `haystack` with optional
//! case-insensitive mode. Returns count and positions.
//!
//! Demonstrates the **TUI.80** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU grep `-c` count flag; vim search-and-count
//!  conventions.
//!
//! Run with: cargo run --example tui_text_search_match_count
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SearchVerdict {
    Ok { count: u32, positions: Vec<u32> },
    InvalidConfig,
}

pub fn search(haystack: &str, needle: &str, case_insensitive: bool) -> SearchVerdict {
    if needle.is_empty() {
        return SearchVerdict::InvalidConfig;
    }
    let (h, n) = if case_insensitive {
        (haystack.to_lowercase(), needle.to_lowercase())
    } else {
        (haystack.to_string(), needle.to_string())
    };
    let mut positions: Vec<u32> = Vec::new();
    let mut start = 0usize;
    while let Some(pos) = h[start..].find(&n) {
        let abs_pos = start + pos;
        positions.push(abs_pos as u32);
        start = abs_pos + n.len().max(1);
    }
    SearchVerdict::Ok {
        count: positions.len() as u32,
        positions,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_text_search_match_count")?;

    println!("simple: {:?}", search("ababab", "ab", false));
    println!(
        "case-insensitive: {:?}",
        search("Hello world Hello", "hello", true)
    );
    println!("invalid: {:?}", search("abc", "", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn three_matches_found() {
        let v = search("ababab", "ab", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn no_match_zero_count() {
        let v = search("foo", "bar", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn case_sensitive_misses_capital() {
        let v = search("Hello", "hello", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn case_insensitive_finds_capital() {
        let v = search("Hello world Hello", "hello", true);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn empty_needle_rejected() {
        assert_eq!(search("abc", "", false), SearchVerdict::InvalidConfig);
    }

    #[test]
    fn positions_correct() {
        let v = search("foo bar foo", "foo", false);
        if let SearchVerdict::Ok { positions, .. } = v {
            assert_eq!(positions, vec![0, 8]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = search("aaa", "a", false);
        let r2 = search("aaa", "a", false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_haystack_zero_matches() {
        let v = search("", "x", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn unicode_search_works() {
        let v = search("café", "é", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn non_overlapping_match_count() {
        // "aaa" / "aa" → only 1 non-overlapping match.
        let v = search("aaa", "aa", false);
        if let SearchVerdict::Ok { count, .. } = v {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn count_matches_positions_len() {
        let v = search("abc abc abc", "abc", false);
        if let SearchVerdict::Ok { count, positions } = v {
            assert_eq!(count, positions.len() as u32);
        }
    }
}
