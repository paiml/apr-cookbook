//! # TUI Search-Replace Preview
//!
//! Preview a search-and-replace operation: render text with all
//! occurrences of `needle` replaced by `replacement`, return both
//! preview text and replace count.
//!
//! Demonstrates the **TUI.101** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:%s/foo/bar/g` substitution; sed s command.
//!
//! Run with: cargo run --example tui_search_replace_preview
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PreviewVerdict {
    Ok { preview: String, replace_count: u32 },
    InvalidConfig,
}

pub fn preview(text: &str, needle: &str, replacement: &str) -> PreviewVerdict {
    if needle.is_empty() {
        return PreviewVerdict::InvalidConfig;
    }
    let preview = text.replace(needle, replacement);
    let replace_count = text.matches(needle).count() as u32;
    PreviewVerdict::Ok {
        preview,
        replace_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_search_replace_preview")?;

    println!("simple: {:?}", preview("foo bar foo baz", "foo", "qux"));
    println!("no match: {:?}", preview("hello world", "xyz", "qux"));
    println!("invalid: {:?}", preview("abc", "", "x"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previewer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn replaces_all_occurrences() {
        let v = preview("foo bar foo", "foo", "qux");
        if let PreviewVerdict::Ok { preview, .. } = v {
            assert_eq!(preview, "qux bar qux");
        }
    }

    #[test]
    fn replace_count_correct() {
        let v = preview("foo foo foo", "foo", "x");
        if let PreviewVerdict::Ok { replace_count, .. } = v {
            assert_eq!(replace_count, 3);
        }
    }

    #[test]
    fn no_match_zero_count() {
        let v = preview("hello", "xyz", "abc");
        if let PreviewVerdict::Ok { replace_count, .. } = v {
            assert_eq!(replace_count, 0);
        }
    }

    #[test]
    fn empty_needle_rejected() {
        assert_eq!(preview("abc", "", "x"), PreviewVerdict::InvalidConfig);
    }

    #[test]
    fn empty_replacement_deletes() {
        let v = preview("hello world", "world", "");
        if let PreviewVerdict::Ok { preview, .. } = v {
            assert_eq!(preview, "hello ");
        }
    }

    #[test]
    fn empty_text_works() {
        let v = preview("", "abc", "x");
        if let PreviewVerdict::Ok {
            preview,
            replace_count,
        } = v
        {
            assert_eq!(preview, "");
            assert_eq!(replace_count, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = preview("hello", "l", "L");
        let r2 = preview("hello", "l", "L");
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let v = preview("Foo foo", "foo", "X");
        if let PreviewVerdict::Ok {
            preview,
            replace_count,
        } = v
        {
            assert_eq!(preview, "Foo X");
            assert_eq!(replace_count, 1);
        }
    }

    #[test]
    fn unicode_replace_works() {
        let v = preview("café", "é", "e");
        if let PreviewVerdict::Ok { preview, .. } = v {
            assert_eq!(preview, "cafe");
        }
    }

    #[test]
    fn longer_replacement_grows_text() {
        let v = preview("abc", "b", "BBB");
        if let PreviewVerdict::Ok { preview, .. } = v {
            assert_eq!(preview, "aBBBc");
        }
    }

    #[test]
    fn shorter_replacement_shrinks_text() {
        let v = preview("hello", "ello", "i");
        if let PreviewVerdict::Ok { preview, .. } = v {
            assert_eq!(preview, "hi");
        }
    }
}
