//! # Contracts-Macros YAML Path Normalize
//!
//! Normalize file paths in YAML config: strip trailing slashes,
//! expand `~/` to home placeholder, collapse `//`. Returns each
//! path's normalized form and changed-flag.
//!
//! Demonstrates the **CMM.128** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: POSIX path canonicalization (realpath); shell tilde
//!  expansion conventions.
//!
//! Run with: cargo run --example contracts_macros_yaml_path_normalize
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NormalizeVerdict {
    Ok {
        normalized: Vec<(String, bool)>,
        changed_count: u32,
    },
    InvalidConfig,
}

pub fn normalize(paths: &[&str], home_placeholder: &str) -> NormalizeVerdict {
    if paths.is_empty() {
        return NormalizeVerdict::InvalidConfig;
    }
    let mut normalized: Vec<(String, bool)> = Vec::with_capacity(paths.len());
    let mut changed_count = 0u32;
    for path in paths {
        let mut s = (*path).to_string();
        // Expand `~/` prefix.
        if let Some(rest) = s.strip_prefix("~/") {
            s = format!("{home_placeholder}/{rest}");
        }
        // Collapse multiple slashes.
        while s.contains("//") {
            s = s.replace("//", "/");
        }
        // Strip trailing slash unless root.
        if s.len() > 1 && s.ends_with('/') {
            s.pop();
        }
        let changed = s != *path;
        if changed {
            changed_count += 1;
        }
        normalized.push((s, changed));
    }
    NormalizeVerdict::Ok {
        normalized,
        changed_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_path_normalize")?;

    let paths = ["/etc/hosts", "/etc/", "~/notes", "/path//to/file"];
    println!("normalize: {:?}", normalize(&paths, "/home/user"));
    println!("invalid: {:?}", normalize(&[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn already_canonical_unchanged() {
        let v = normalize(&["/etc/hosts"], "/home/user");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/etc/hosts");
            assert!(!normalized[0].1);
        }
    }

    #[test]
    fn trailing_slash_stripped() {
        let v = normalize(&["/etc/"], "/home/user");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/etc");
        }
    }

    #[test]
    fn tilde_expanded() {
        let v = normalize(&["~/notes"], "/home/user");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/home/user/notes");
        }
    }

    #[test]
    fn double_slash_collapsed() {
        let v = normalize(&["/path//to/file"], "/home/user");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/path/to/file");
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(normalize(&[], "x"), NormalizeVerdict::InvalidConfig);
    }

    #[test]
    fn root_slash_preserved() {
        let v = normalize(&["/"], "/home");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/");
        }
    }

    #[test]
    fn changed_count_correct() {
        let v = normalize(&["/ok", "/with/", "/already/clean"], "/home");
        if let NormalizeVerdict::Ok { changed_count, .. } = v {
            assert_eq!(changed_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = normalize(&["~/a"], "/home/u");
        let r2 = normalize(&["~/a"], "/home/u");
        assert_eq!(r1, r2);
    }

    #[test]
    fn triple_slash_collapsed() {
        let v = normalize(&["/a///b"], "/home");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/a/b");
        }
    }

    #[test]
    fn relative_path_left_alone() {
        let v = normalize(&["relative/path"], "/home");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "relative/path");
        }
    }

    #[test]
    fn unicode_path_supported() {
        let v = normalize(&["/café/"], "/home");
        if let NormalizeVerdict::Ok { normalized, .. } = v {
            assert_eq!(normalized[0].0, "/café");
        }
    }
}
