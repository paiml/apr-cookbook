//! # Contracts-Macros Recipe Link Health
//!
//! Cross-reference recipe citations against a known-good DOI/arXiv
//! list. Returns broken links and per-link verdicts (KnownGood,
//! BadFormat, Unknown).
//!
//! Demonstrates the **CMM.90** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: DOI handle resolution conventions (RFC 3986); arXiv
//!  identifier scheme (Cornell University Library, 1991).
//!
//! Run with: cargo run --example contracts_macros_recipe_link_health
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Clone)]
pub enum LinkStatus {
    KnownGood,
    BadFormat,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum LinkVerdict {
    Ok {
        per_link: Vec<(String, LinkStatus)>,
        good_count: u32,
    },
    InvalidConfig,
}

pub fn audit(citations: &[&str], known_good: &[&str]) -> LinkVerdict {
    if citations.is_empty() {
        return LinkVerdict::InvalidConfig;
    }
    let known_set: BTreeSet<&str> = known_good.iter().copied().collect();
    let mut per_link: Vec<(String, LinkStatus)> = Vec::with_capacity(citations.len());
    let mut good_count = 0u32;
    for cite in citations {
        let status = classify(cite, &known_set);
        if status == LinkStatus::KnownGood {
            good_count += 1;
        }
        per_link.push(((*cite).to_string(), status));
    }
    LinkVerdict::Ok {
        per_link,
        good_count,
    }
}

fn classify(cite: &str, known_set: &BTreeSet<&str>) -> LinkStatus {
    if known_set.contains(cite) {
        return LinkStatus::KnownGood;
    }
    // Validate format: must be doi: or arXiv: or starts with 10. (raw DOI).
    let valid = cite.starts_with("doi:") || cite.starts_with("arXiv:") || cite.starts_with("10.");
    if valid {
        LinkStatus::Unknown
    } else {
        LinkStatus::BadFormat
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_link_health")?;

    let citations = [
        "doi:10.1145/12345.67890",
        "arXiv:1706.03762",
        "not_a_link",
        "10.1109/foo",
    ];
    let known = ["doi:10.1145/12345.67890"];
    println!("audit: {:?}", audit(&citations, &known));
    println!("invalid: {:?}", audit(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_doi_recognized() {
        let v = audit(&["doi:10.1145/x"], &["doi:10.1145/x"]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::KnownGood);
        }
    }

    #[test]
    fn arxiv_format_unknown_when_not_in_list() {
        let v = audit(&["arXiv:1706.03762"], &[]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::Unknown);
        }
    }

    #[test]
    fn bad_format_flagged() {
        let v = audit(&["not_a_link"], &[]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::BadFormat);
        }
    }

    #[test]
    fn empty_citations_rejected() {
        assert_eq!(audit(&[], &[]), LinkVerdict::InvalidConfig);
    }

    #[test]
    fn good_count_correct() {
        let v = audit(&["doi:1", "doi:2", "bad"], &["doi:1", "doi:2"]);
        if let LinkVerdict::Ok { good_count, .. } = v {
            assert_eq!(good_count, 2);
        }
    }

    #[test]
    fn raw_doi_format_accepted() {
        let v = audit(&["10.1109/foo"], &[]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::Unknown);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["doi:1"], &["doi:1"]);
        let r2 = audit(&["doi:1"], &["doi:1"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_matches_input_length() {
        let v = audit(&["a", "b", "c"], &[]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link.len(), 3);
        }
    }

    #[test]
    fn case_sensitive_known_match() {
        let v = audit(&["DOI:1"], &["doi:1"]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_ne!(per_link[0].1, LinkStatus::KnownGood);
        }
    }

    #[test]
    fn three_categories_present() {
        let v = audit(&["doi:1", "arXiv:x", "garbage"], &["doi:1"]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::KnownGood);
            assert_eq!(per_link[1].1, LinkStatus::Unknown);
            assert_eq!(per_link[2].1, LinkStatus::BadFormat);
        }
    }

    #[test]
    fn empty_known_good_list_works() {
        let v = audit(&["doi:x"], &[]);
        if let LinkVerdict::Ok { per_link, .. } = v {
            assert_eq!(per_link[0].1, LinkStatus::Unknown);
        }
    }
}
