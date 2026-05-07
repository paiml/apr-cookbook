//! # TUI Spell Check Underline
//!
//! Mark words not in the dictionary for underline rendering. Returns
//! `(word, start_offset)` pairs sorted by offset.
//!
//! Demonstrates the **TUI.136** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:set spell` underline rendering; emacs `flyspell`
//!  on-the-fly mark-up.
//!
//! Run with: cargo run --example tui_spell_check_underline
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SpellVerdict {
    Ok {
        misspellings: Vec<(String, u32)>,
        word_count: u32,
    },
    InvalidConfig,
}

pub fn check(text: &str, dictionary: &[&str]) -> SpellVerdict {
    if text.is_empty() {
        return SpellVerdict::InvalidConfig;
    }
    let dict: BTreeSet<String> = dictionary.iter().map(|s| s.to_lowercase()).collect();
    let mut misspellings: Vec<(String, u32)> = Vec::new();
    let mut word_count = 0u32;
    let mut start = 0usize;
    while start < text.len() {
        let bytes = text.as_bytes();
        if !bytes[start].is_ascii_alphabetic() {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < text.len() && text.as_bytes()[end].is_ascii_alphabetic() {
            end += 1;
        }
        let word = &text[start..end];
        word_count += 1;
        if !dict.contains(&word.to_lowercase()) {
            misspellings.push((word.to_string(), start as u32));
        }
        start = end;
    }
    SpellVerdict::Ok {
        misspellings,
        word_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_spell_check_underline")?;

    let dict = ["the", "quick", "brown", "fox"];
    println!("check: {:?}", check("the quikc brown fox", &dict));
    println!("invalid: {:?}", check("", &dict));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_words_known_no_misspellings() {
        let v = check("the fox", &["the", "fox"]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert!(misspellings.is_empty());
        }
    }

    #[test]
    fn unknown_word_flagged() {
        let v = check("xyz", &["the"]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert_eq!(misspellings.len(), 1);
            assert_eq!(misspellings[0].0, "xyz");
        }
    }

    #[test]
    fn offset_correct() {
        let v = check("the xyz", &["the"]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert_eq!(misspellings[0].1, 4);
        }
    }

    #[test]
    fn case_insensitive_dict() {
        let v = check("THE", &["the"]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert!(misspellings.is_empty());
        }
    }

    #[test]
    fn empty_text_rejected() {
        assert_eq!(check("", &["the"]), SpellVerdict::InvalidConfig);
    }

    #[test]
    fn punctuation_word_split() {
        let v = check("hi, world", &["hi", "world"]);
        if let SpellVerdict::Ok { word_count, .. } = v {
            assert_eq!(word_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("hello", &["hi"]);
        let r2 = check("hello", &["hi"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn word_count_correct() {
        let v = check("a b c d", &["a", "b", "c", "d"]);
        if let SpellVerdict::Ok { word_count, .. } = v {
            assert_eq!(word_count, 4);
        }
    }

    #[test]
    fn no_dict_all_flagged() {
        let v = check("hello world", &[]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert_eq!(misspellings.len(), 2);
        }
    }

    #[test]
    fn offsets_ascending() {
        let v = check("xyz abc def", &[]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            for w in misspellings.windows(2) {
                assert!(w[0].1 < w[1].1);
            }
        }
    }

    #[test]
    fn many_words_handled() {
        let text: String = (0..30).map(|_| "xyz ").collect();
        let v = check(&text, &[]);
        if let SpellVerdict::Ok { misspellings, .. } = v {
            assert_eq!(misspellings.len(), 30);
        }
    }

    #[test]
    fn digits_skipped() {
        let v = check("hi 123 world", &["hi", "world"]);
        if let SpellVerdict::Ok { word_count, .. } = v {
            assert_eq!(word_count, 2);
        }
    }
}
