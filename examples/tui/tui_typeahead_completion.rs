//! # TUI Typeahead Completion
//!
//! Suggest completions for partial input from a sorted word list.
//! Returns up to `max_results` matches sorted alphabetically.
//!
//! Demonstrates the **TUI.99** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: bash readline tab-completion; rlwrap typeahead history.
//!
//! Run with: cargo run --example tui_typeahead_completion
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompletionVerdict {
    Ok {
        matches: Vec<String>,
        truncated: bool,
    },
    InvalidConfig,
}

pub fn complete(words: &[&str], prefix: &str, max_results: u32) -> CompletionVerdict {
    if words.is_empty() || max_results == 0 {
        return CompletionVerdict::InvalidConfig;
    }
    let p_lower = prefix.to_lowercase();
    let mut matches: Vec<String> = words
        .iter()
        .filter(|w| w.to_lowercase().starts_with(&p_lower))
        .map(|w| (*w).to_string())
        .collect();
    matches.sort();
    matches.dedup();
    let truncated = matches.len() as u32 > max_results;
    matches.truncate(max_results as usize);
    CompletionVerdict::Ok { matches, truncated }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_typeahead_completion")?;

    let words = ["apple", "apricot", "banana", "cherry", "avocado"];
    println!("prefix 'a': {:?}", complete(&words, "a", 10));
    println!("prefix 'ap': {:?}", complete(&words, "ap", 10));
    println!("invalid: {:?}", complete(&[], "a", 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn prefix_matches_returned() {
        let words = ["apple", "apricot", "banana"];
        let v = complete(&words, "ap", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches.len(), 2);
        }
    }

    #[test]
    fn no_match_empty() {
        let words = ["apple", "banana"];
        let v = complete(&words, "xyz", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert!(matches.is_empty());
        }
    }

    #[test]
    fn case_insensitive_prefix() {
        let words = ["Apple"];
        let v = complete(&words, "AP", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches, vec!["Apple".to_string()]);
        }
    }

    #[test]
    fn empty_words_rejected() {
        assert_eq!(complete(&[], "a", 10), CompletionVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        let words = ["a"];
        assert_eq!(complete(&words, "a", 0), CompletionVerdict::InvalidConfig);
    }

    #[test]
    fn max_results_truncates() {
        let words = ["apple", "apricot", "avocado"];
        let v = complete(&words, "a", 2);
        if let CompletionVerdict::Ok { matches, truncated } = v {
            assert_eq!(matches.len(), 2);
            assert!(truncated);
        }
    }

    #[test]
    fn empty_prefix_returns_all() {
        let words = ["apple", "banana"];
        let v = complete(&words, "", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches.len(), 2);
        }
    }

    #[test]
    fn matches_sorted() {
        let words = ["zeta", "apple", "banana"];
        let v = complete(&words, "", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches[0], "apple");
            assert_eq!(matches[1], "banana");
        }
    }

    #[test]
    fn deterministic() {
        let words = ["a", "b"];
        let r1 = complete(&words, "a", 10);
        let r2 = complete(&words, "a", 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn duplicate_words_dedup() {
        let words = ["apple", "apple", "banana"];
        let v = complete(&words, "a", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches.len(), 1);
        }
    }

    #[test]
    fn unicode_prefix_supported() {
        let words = ["café", "résumé"];
        let v = complete(&words, "café", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches, vec!["café".to_string()]);
        }
    }

    #[test]
    fn full_word_self_match() {
        let words = ["apple"];
        let v = complete(&words, "apple", 10);
        if let CompletionVerdict::Ok { matches, .. } = v {
            assert_eq!(matches, vec!["apple".to_string()]);
        }
    }
}
