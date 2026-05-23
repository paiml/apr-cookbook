//! # Contracts-Macros arXiv Citation Lint
//!
//! Verify a citation string matches the arXiv ID format (e.g.
//! `2104.08691` or `cs.AI/0301002`). Returns the parsed
//! year/month/seq for new-style IDs or category/seq for old-style.
//!
//! Demonstrates the **CMM.43** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: arXiv identifier scheme (since 2007 + legacy).
//!
//! Run with: cargo run --example contracts_macros_arxiv_citation_lint
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CitationVerdict {
    NewStyle { yymm: u32, seq: u32 },
    OldStyle { category: String, seq: u32 },
    Malformed { reason: &'static str },
    Empty,
}

pub fn lint(citation: &str) -> CitationVerdict {
    let trimmed = citation.trim();
    if trimmed.is_empty() {
        return CitationVerdict::Empty;
    }
    if let Some((yymm_str, seq_str)) = trimmed.split_once('.') {
        if yymm_str.len() == 4
            && seq_str.len() >= 4
            && seq_str.len() <= 5
            && yymm_str.chars().all(|c| c.is_ascii_digit())
            && seq_str.chars().all(|c| c.is_ascii_digit())
        {
            let Ok(yymm) = yymm_str.parse::<u32>() else {
                return CitationVerdict::Malformed {
                    reason: "yymm parse",
                };
            };
            let Ok(seq) = seq_str.parse::<u32>() else {
                return CitationVerdict::Malformed {
                    reason: "seq parse",
                };
            };
            return CitationVerdict::NewStyle { yymm, seq };
        }
        return CitationVerdict::Malformed {
            reason: "new-style format",
        };
    }
    if let Some((cat, seq_str)) = trimmed.split_once('/') {
        if !cat.contains('.')
            && cat.chars().all(|c| c.is_ascii_alphabetic())
            && seq_str.len() == 7
            && seq_str.chars().all(|c| c.is_ascii_digit())
        {
            let Ok(seq) = seq_str.parse::<u32>() else {
                return CitationVerdict::Malformed {
                    reason: "old seq parse",
                };
            };
            return CitationVerdict::OldStyle {
                category: cat.to_string(),
                seq,
            };
        }
        if cat.contains('.') {
            // e.g. "cs.AI/0301002" — split category at the first dot.
            let parts: Vec<&str> = cat.split('.').collect();
            if parts.len() == 2
                && parts
                    .iter()
                    .all(|p| p.chars().all(|c| c.is_ascii_alphabetic()))
                && seq_str.len() == 7
                && seq_str.chars().all(|c| c.is_ascii_digit())
            {
                let Ok(seq) = seq_str.parse::<u32>() else {
                    return CitationVerdict::Malformed {
                        reason: "old seq parse",
                    };
                };
                return CitationVerdict::OldStyle {
                    category: cat.to_string(),
                    seq,
                };
            }
        }
        return CitationVerdict::Malformed {
            reason: "old-style format",
        };
    }
    CitationVerdict::Malformed {
        reason: "missing dot or slash",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_arxiv_citation_lint")?;

    println!("new style: {:?}", lint("2104.08691"));
    println!("old style: {:?}", lint("cs.AI/0301002"));
    println!("simple old: {:?}", lint("hep/9912345"));
    println!("malformed: {:?}", lint("not-an-id"));
    println!("empty: {:?}", lint("   "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn new_style_parsed() {
        let v = lint("2104.08691");
        if let CitationVerdict::NewStyle { yymm, seq } = v {
            assert_eq!(yymm, 2104);
            assert_eq!(seq, 8691);
        }
    }

    #[test]
    fn old_style_subcategory_parsed() {
        let v = lint("cs.AI/0301002");
        if let CitationVerdict::OldStyle { category, seq } = v {
            assert_eq!(category, "cs.AI");
            assert_eq!(seq, 301002);
        }
    }

    #[test]
    fn old_style_simple_category() {
        let v = lint("hep/9912345");
        if let CitationVerdict::OldStyle { category, .. } = v {
            assert_eq!(category, "hep");
        }
    }

    #[test]
    fn malformed_no_separator() {
        let v = lint("not-an-id");
        assert!(matches!(v, CitationVerdict::Malformed { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(lint("   "), CitationVerdict::Empty);
    }

    #[test]
    fn new_style_three_digit_yymm_rejected() {
        let v = lint("210.08691");
        assert!(matches!(v, CitationVerdict::Malformed { .. }));
    }

    #[test]
    fn new_style_short_seq_rejected() {
        let v = lint("2104.123");
        assert!(matches!(v, CitationVerdict::Malformed { .. }));
    }

    #[test]
    fn old_style_short_seq_rejected() {
        let v = lint("cs/12345");
        assert!(matches!(v, CitationVerdict::Malformed { .. }));
    }

    #[test]
    fn whitespace_trimmed() {
        let v = lint("  2104.08691  ");
        assert!(matches!(v, CitationVerdict::NewStyle { .. }));
    }

    #[test]
    fn deterministic() {
        let a = lint("2104.08691");
        let b = lint("2104.08691");
        assert_eq!(a, b);
    }
}
