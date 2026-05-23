//! # TUI Breadcrumb Search Filter
//!
//! Filter breadcrumb segments by case-insensitive substring query.
//! Returns matched segment indices and whether the path still has
//! contiguous filter coverage.
//!
//! Demonstrates the **TUI.90** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code breadcrumb search; macOS Finder filter bar.
//!
//! Run with: cargo run --example tui_breadcrumb_search_filter
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok {
        matched_indices: Vec<u32>,
        any_match: bool,
    },
    InvalidConfig,
}

pub fn filter(segments: &[&str], query: &str) -> FilterVerdict {
    if segments.is_empty() || query.is_empty() {
        return FilterVerdict::InvalidConfig;
    }
    let q_lower = query.to_lowercase();
    let mut matched_indices: Vec<u32> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if seg.to_lowercase().contains(&q_lower) {
            matched_indices.push(i as u32);
        }
    }
    let any_match = !matched_indices.is_empty();
    FilterVerdict::Ok {
        matched_indices,
        any_match,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_breadcrumb_search_filter")?;

    let segments = ["Home", "Documents", "Code", "Projects", "Demo"];
    println!("query 'doc': {:?}", filter(&segments, "doc"));
    println!("query 'xyz': {:?}", filter(&segments, "xyz"));
    println!("invalid: {:?}", filter(&[], "doc"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn substring_match_found() {
        let v = filter(&["alpha", "beta", "gamma"], "ph");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0]);
        }
    }

    #[test]
    fn no_match_returns_empty() {
        let v = filter(&["alpha", "beta"], "zzz");
        if let FilterVerdict::Ok {
            matched_indices,
            any_match,
        } = v
        {
            assert!(matched_indices.is_empty());
            assert!(!any_match);
        }
    }

    #[test]
    fn empty_segments_rejected() {
        assert_eq!(filter(&[], "a"), FilterVerdict::InvalidConfig);
    }

    #[test]
    fn empty_query_rejected() {
        let segs = ["a"];
        assert_eq!(filter(&segs, ""), FilterVerdict::InvalidConfig);
    }

    #[test]
    fn case_insensitive() {
        let v = filter(&["Alpha"], "ALPHA");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0]);
        }
    }

    #[test]
    fn multiple_matches_collected() {
        let v = filter(&["alpha", "alphabet", "beta"], "alpha");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0, 1]);
        }
    }

    #[test]
    fn any_match_true_when_found() {
        let v = filter(&["a"], "a");
        if let FilterVerdict::Ok { any_match, .. } = v {
            assert!(any_match);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = filter(&["alpha"], "a");
        let r2 = filter(&["alpha"], "a");
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_query_supported() {
        let v = filter(&["café", "résumé"], "café");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0]);
        }
    }

    #[test]
    fn full_segment_match() {
        let v = filter(&["alpha"], "alpha");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0]);
        }
    }

    #[test]
    fn substring_not_full_match() {
        let v = filter(&["alpha", "beta"], "lph");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0]);
        }
    }

    #[test]
    fn matched_indices_in_order() {
        let v = filter(&["a-x", "b-x", "c-x"], "x");
        if let FilterVerdict::Ok {
            matched_indices, ..
        } = v
        {
            assert_eq!(matched_indices, vec![0, 1, 2]);
        }
    }
}
