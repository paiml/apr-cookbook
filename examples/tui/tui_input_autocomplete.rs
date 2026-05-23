//! # TUI Input Autocomplete
//!
//! Suggest completions for the current input prefix from a vocabulary.
//! Returns top-K matches, ranked by length (shorter first) then alpha.
//!
//! Demonstrates the **TUI.50** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: bash tab-complete + IDE prefix-search behaviour.
//!
//! Run with: cargo run --example tui_input_autocomplete
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AutocompleteVerdict {
    Ok { suggestions: Vec<String> },
    EmptyVocabulary,
    NoPrefix,
}

pub fn suggest(prefix: &str, vocabulary: &[&str], top_k: u32) -> AutocompleteVerdict {
    if vocabulary.is_empty() {
        return AutocompleteVerdict::EmptyVocabulary;
    }
    if prefix.is_empty() {
        return AutocompleteVerdict::NoPrefix;
    }
    let mut matches: Vec<&str> = vocabulary
        .iter()
        .filter(|w| w.starts_with(prefix))
        .copied()
        .collect();
    matches.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    let suggestions: Vec<String> = matches
        .into_iter()
        .take(top_k as usize)
        .map(String::from)
        .collect();
    AutocompleteVerdict::Ok { suggestions }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_input_autocomplete")?;

    let vocab = ["open", "open_file", "open_folder", "save", "search"];
    println!("'op': {:?}", suggest("op", &vocab, 5));
    println!("'sa': {:?}", suggest("sa", &vocab, 5));
    println!("'xy': {:?}", suggest("xy", &vocab, 5));
    println!("empty prefix: {:?}", suggest("", &vocab, 5));
    println!("empty vocab: {:?}", suggest("op", &[], 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vec<&'static str> {
        vec!["open", "open_file", "open_folder", "save", "search"]
    }

    #[test]
    fn suggester_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn prefix_match() {
        let v = suggest("op", &vocab(), 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert_eq!(suggestions.len(), 3);
        }
    }

    #[test]
    fn shorter_first() {
        let v = suggest("op", &vocab(), 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert_eq!(suggestions[0], "open");
        }
    }

    #[test]
    fn no_match_empty_list() {
        let v = suggest("xy", &vocab(), 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert!(suggestions.is_empty());
        }
    }

    #[test]
    fn empty_prefix_rejected() {
        assert_eq!(suggest("", &vocab(), 5), AutocompleteVerdict::NoPrefix);
    }

    #[test]
    fn empty_vocab_rejected() {
        assert_eq!(suggest("op", &[], 5), AutocompleteVerdict::EmptyVocabulary);
    }

    #[test]
    fn top_k_limits_results() {
        let v = suggest("op", &vocab(), 1);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert_eq!(suggestions.len(), 1);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = suggest("OP", &vocab(), 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert!(suggestions.is_empty());
        }
    }

    #[test]
    fn alpha_tiebreak_within_length() {
        let v = suggest("o", &["oz", "ox", "oa"], 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert_eq!(suggestions[0], "oa");
        }
    }

    #[test]
    fn full_word_match() {
        let v = suggest("save", &vocab(), 5);
        if let AutocompleteVerdict::Ok { suggestions } = v {
            assert!(suggestions.iter().any(|s| s == "save"));
        }
    }

    #[test]
    fn deterministic() {
        let v = vocab();
        let a = suggest("op", &v, 5);
        let b = suggest("op", &v, 5);
        assert_eq!(a, b);
    }
}
