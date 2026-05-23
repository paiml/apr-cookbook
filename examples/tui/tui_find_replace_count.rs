//! # TUI Find-and-Replace Match Count
//!
//! Count substring matches of `needle` in `haystack`, optionally
//! whole-word and case-insensitive. Returns match count + first
//! match byte offset (or `u32::MAX` if none).
//!
//! Demonstrates the **TUI.132** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:s/.../.../gn` count syntax; sed match-only mode.
//!
//! Run with: cargo run --example tui_find_replace_count
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FindVerdict {
    Ok { count: u32, first_offset: u32 },
    InvalidConfig,
}

pub fn find_count(
    haystack: &str,
    needle: &str,
    case_insensitive: bool,
    whole_word: bool,
) -> FindVerdict {
    if needle.is_empty() {
        return FindVerdict::InvalidConfig;
    }
    let h: String = if case_insensitive {
        haystack.to_lowercase()
    } else {
        haystack.to_string()
    };
    let n: String = if case_insensitive {
        needle.to_lowercase()
    } else {
        needle.to_string()
    };
    let mut count = 0u32;
    let mut first = u32::MAX;
    let mut start = 0usize;
    while let Some(idx) = h[start..].find(&n) {
        let abs = start + idx;
        let after = abs + n.len();
        let valid = if whole_word {
            let before_ok = abs == 0 || !h.as_bytes()[abs - 1].is_ascii_alphanumeric();
            let after_ok = after >= h.len() || !h.as_bytes()[after].is_ascii_alphanumeric();
            before_ok && after_ok
        } else {
            true
        };
        if valid {
            count += 1;
            if first == u32::MAX {
                first = abs as u32;
            }
        }
        start = abs + 1;
    }
    FindVerdict::Ok {
        count,
        first_offset: first,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_find_replace_count")?;

    println!(
        "count 'foo': {:?}",
        find_count("foo bar foo baz foo", "foo", false, false)
    );
    println!(
        "case-insens: {:?}",
        find_count("Foo foo FOO", "foo", true, false)
    );
    println!("invalid: {:?}", find_count("x", "", false, false));
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
    fn three_matches_counted() {
        let v = find_count("foo bar foo baz foo", "foo", false, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn no_match_zero_count() {
        let v = find_count("abc", "xyz", false, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn first_offset_correct() {
        let v = find_count("xxxfooxxx", "foo", false, false);
        if let FindVerdict::Ok { first_offset, .. } = v {
            assert_eq!(first_offset, 3);
        }
    }

    #[test]
    fn empty_needle_invalid() {
        assert_eq!(
            find_count("abc", "", false, false),
            FindVerdict::InvalidConfig
        );
    }

    #[test]
    fn case_insensitive_match() {
        let v = find_count("Foo FOO foo", "foo", true, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn case_sensitive_mismatch() {
        let v = find_count("Foo FOO", "foo", false, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn whole_word_excludes_partial() {
        // "foo" inside "fooz" not whole-word
        let v = find_count("foo fooz foo", "foo", false, true);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn whole_word_inclusive_at_boundaries() {
        let v = find_count("foo. bar; foo!", "foo", false, true);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = find_count("hello", "ll", false, false);
        let r2 = find_count("hello", "ll", false, false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_match_offset_max() {
        let v = find_count("abc", "xyz", false, false);
        if let FindVerdict::Ok { first_offset, .. } = v {
            assert_eq!(first_offset, u32::MAX);
        }
    }

    #[test]
    fn overlapping_matches_counted() {
        // "aa" in "aaaa" matches at 0,1,2 (overlapping)
        let v = find_count("aaaa", "aa", false, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn unicode_haystack_supported() {
        let v = find_count("café café", "café", false, false);
        if let FindVerdict::Ok { count, .. } = v {
            assert_eq!(count, 2);
        }
    }
}
