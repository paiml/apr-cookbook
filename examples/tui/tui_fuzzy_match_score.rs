//! # TUI Fuzzy Match Score
//!
//! Score how well a query matches a candidate string using a simple
//! fuzzy-matching algorithm: each query char must appear in order
//! in the candidate. Returns score (0=no match, higher=better) and
//! match positions.
//!
//! Demonstrates the **TUI.180** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: fzf scoring algorithm; Sublime Text command-palette
//!  fuzzy match.
//!
//! Run with: cargo run --example tui_fuzzy_match_score
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FuzzyVerdict {
    Ok { score: u32, positions: Vec<u32> },
    NoMatch,
    InvalidConfig,
}

pub fn score(query: &str, candidate: &str) -> FuzzyVerdict {
    if query.is_empty() || candidate.is_empty() {
        return FuzzyVerdict::InvalidConfig;
    }
    let cand_lower = candidate.to_lowercase();
    let query_lower = query.to_lowercase();
    let cand_chars: Vec<char> = cand_lower.chars().collect();
    let query_chars: Vec<char> = query_lower.chars().collect();
    let mut positions: Vec<u32> = Vec::new();
    let mut q_idx = 0usize;
    for (c_idx, c) in cand_chars.iter().enumerate() {
        if q_idx < query_chars.len() && *c == query_chars[q_idx] {
            positions.push(c_idx as u32);
            q_idx += 1;
        }
    }
    if q_idx < query_chars.len() {
        return FuzzyVerdict::NoMatch;
    }
    // Score: prefer earlier positions and consecutive runs.
    let mut s: u32 = 100;
    if !positions.is_empty() {
        s = s.saturating_sub(positions[0]);
        for w in positions.windows(2) {
            if w[1] - w[0] == 1 {
                s = s.saturating_add(2); // bonus for consecutive
            }
        }
    }
    FuzzyVerdict::Ok {
        score: s,
        positions,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_fuzzy_match_score")?;

    println!("match: {:?}", score("rs", "rust_recipe.rs"));
    println!("no-match: {:?}", score("zzz", "abc"));
    println!("invalid: {:?}", score("", "x"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_query_rejected() {
        assert_eq!(score("", "abc"), FuzzyVerdict::InvalidConfig);
    }

    #[test]
    fn empty_candidate_rejected() {
        assert_eq!(score("abc", ""), FuzzyVerdict::InvalidConfig);
    }

    #[test]
    fn full_match_returns_score() {
        let v = score("rs", "rust");
        if let FuzzyVerdict::Ok { positions, .. } = v {
            assert_eq!(positions.len(), 2);
        }
    }

    #[test]
    fn no_match_returns_nomatch() {
        assert_eq!(score("xyz", "abc"), FuzzyVerdict::NoMatch);
    }

    #[test]
    fn case_insensitive() {
        let v = score("RS", "rust");
        if let FuzzyVerdict::Ok { positions, .. } = v {
            assert_eq!(positions.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = score("a", "abc");
        let r2 = score("a", "abc");
        assert_eq!(r1, r2);
    }

    #[test]
    fn positions_ordered() {
        let v = score("rs", "rust_recipe.rs");
        if let FuzzyVerdict::Ok { positions, .. } = v {
            for w in positions.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn earlier_positions_higher_score() {
        let early = score("a", "abc");
        let late = score("a", "xxxa");
        if let (FuzzyVerdict::Ok { score: e, .. }, FuzzyVerdict::Ok { score: l, .. }) =
            (early, late)
        {
            assert!(e > l);
        }
    }

    #[test]
    fn consecutive_run_bonus() {
        let consec = score("ab", "abc");
        let split = score("ac", "abc");
        if let (FuzzyVerdict::Ok { score: c, .. }, FuzzyVerdict::Ok { score: s, .. }) =
            (consec, split)
        {
            assert!(c >= s);
        }
    }

    #[test]
    fn unicode_supported() {
        let v = score("café", "café_function");
        assert!(matches!(v, FuzzyVerdict::Ok { .. }));
    }

    #[test]
    fn out_of_order_no_match() {
        // "ba" cannot match "abc" because b appears after a.
        assert_eq!(score("ba", "abc"), FuzzyVerdict::NoMatch);
    }

    #[test]
    fn long_candidate_handled() {
        let v = score(
            "a",
            &"x".repeat(100).chars().chain(['a']).collect::<String>(),
        );
        assert!(matches!(v, FuzzyVerdict::Ok { .. }));
    }

    #[test]
    fn single_char_match() {
        let v = score("x", "x");
        if let FuzzyVerdict::Ok { positions, .. } = v {
            assert_eq!(positions, vec![0]);
        }
    }
}
