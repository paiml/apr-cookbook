//! # TUI Truncate Path With Dots
//!
//! Truncate file path mid-component with `…` if total length exceeds
//! max_chars. Always preserves first and last components when
//! possible.
//!
//! Demonstrates the **TUI.73** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Finder path bar; tab-completion middle-elision.
//!
//! Run with: cargo run --example tui_truncate_path_with_dots
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PathVerdict {
    Ok { rendered: String, truncated: bool },
    InvalidConfig,
}

pub fn truncate(path: &str, max_chars: u32) -> PathVerdict {
    if max_chars == 0 || path.is_empty() {
        return PathVerdict::InvalidConfig;
    }
    let len = path.chars().count() as u32;
    if len <= max_chars {
        return PathVerdict::Ok {
            rendered: path.to_string(),
            truncated: false,
        };
    }
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        // Can't elide middle, just hard-truncate to max_chars - 1 + "…".
        let chars: Vec<char> = path.chars().take(max_chars as usize - 1).collect();
        let mut s: String = chars.into_iter().collect();
        s.push('…');
        return PathVerdict::Ok {
            rendered: s,
            truncated: true,
        };
    }
    let first = parts.first().unwrap();
    let last = parts.last().unwrap();
    let prefix = if path.starts_with('/') { "/" } else { "" };
    let rendered = format!("{prefix}{first}/…/{last}");
    PathVerdict::Ok {
        rendered,
        truncated: true,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_truncate_path_with_dots")?;

    println!("short: {:?}", truncate("/etc/hosts", 100));
    println!(
        "long: {:?}",
        truncate("/usr/local/share/man/man1/cargo.1", 20)
    );
    println!("invalid: {:?}", truncate("", 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn short_path_unchanged() {
        let v = truncate("/etc/hosts", 100);
        if let PathVerdict::Ok {
            rendered,
            truncated,
        } = v
        {
            assert_eq!(rendered, "/etc/hosts");
            assert!(!truncated);
        }
    }

    #[test]
    fn long_path_elides_middle() {
        let v = truncate("/usr/local/share/man/man1/cargo.1", 20);
        if let PathVerdict::Ok {
            rendered,
            truncated,
        } = v
        {
            assert!(rendered.contains('…'));
            assert!(rendered.starts_with("/usr"));
            assert!(rendered.ends_with("cargo.1"));
            assert!(truncated);
        }
    }

    #[test]
    fn empty_path_rejected() {
        assert_eq!(truncate("", 10), PathVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(truncate("/x", 0), PathVerdict::InvalidConfig);
    }

    #[test]
    fn first_and_last_segments_preserved() {
        let v = truncate("/a/b/c/d", 5);
        if let PathVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('a'));
            assert!(rendered.contains('d'));
        }
    }

    #[test]
    fn relative_path_works() {
        let v = truncate("a/b/c/d/e", 5);
        if let PathVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('a'));
            assert!(rendered.contains('e'));
            assert!(!rendered.starts_with('/'));
        }
    }

    #[test]
    fn single_component_hard_truncated() {
        let v = truncate("verylongname", 5);
        if let PathVerdict::Ok {
            rendered,
            truncated,
        } = v
        {
            assert!(rendered.ends_with('…'));
            assert!(truncated);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = truncate("/a/b/c", 4);
        let r2 = truncate("/a/b/c", 4);
        assert_eq!(r1, r2);
    }

    #[test]
    fn exact_fit_no_truncation() {
        let v = truncate("/etc", 4);
        if let PathVerdict::Ok { truncated, .. } = v {
            assert!(!truncated);
        }
    }

    #[test]
    fn unicode_path_supported() {
        let v = truncate("/café/résumé/x", 100);
        if let PathVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "/café/résumé/x");
        }
    }

    #[test]
    fn rendered_contains_dots_marker() {
        let v = truncate("/a/b/c/d/e", 5);
        if let PathVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('…'));
        }
    }
}
