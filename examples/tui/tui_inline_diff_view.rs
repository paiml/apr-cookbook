//! # TUI Inline Char-Level Diff View
//!
//! Render a char-level diff of two short strings as a sequence of
//! `(verdict, char)` pairs (Same, Insert, Delete) using a simple
//! greedy LCS pass. Useful for inline diff highlighting in TUIs.
//!
//! Demonstrates the **TUI.62** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Myers, E.W., An O(ND) Difference Algorithm (1986).
//!
//! Run with: cargo run --example tui_inline_diff_view
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Op {
    Same,
    Insert,
    Delete,
}

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok { ops: Vec<(Op, char)> },
    InvalidConfig,
}

pub fn diff(a: &str, b: &str, max_len: usize) -> DiffVerdict {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    if av.len() > max_len || bv.len() > max_len {
        return DiffVerdict::InvalidConfig;
    }
    if av.is_empty() && bv.is_empty() {
        return DiffVerdict::InvalidConfig;
    }
    // O(n*m) LCS table.
    let n = av.len();
    let m = bv.len();
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            if av[i] == bv[j] {
                lcs[i + 1][j + 1] = lcs[i][j] + 1;
            } else {
                lcs[i + 1][j + 1] = lcs[i + 1][j].max(lcs[i][j + 1]);
            }
        }
    }
    // Backtrack.
    let mut ops: Vec<(Op, char)> = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && av[i - 1] == bv[j - 1] {
            ops.push((Op::Same, av[i - 1]));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            ops.push((Op::Insert, bv[j - 1]));
            j -= 1;
        } else {
            ops.push((Op::Delete, av[i - 1]));
            i -= 1;
        }
    }
    ops.reverse();
    DiffVerdict::Ok { ops }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_inline_diff_view")?;

    println!("identical: {:?}", diff("hello", "hello", 1024));
    println!("insertion: {:?}", diff("hi", "high", 1024));
    println!("deletion: {:?}", diff("hello", "hell", 1024));
    println!("invalid: {:?}", diff("", "", 1024));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_strings_all_same() {
        let v = diff("abc", "abc", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().all(|(op, _)| *op == Op::Same));
        }
    }

    #[test]
    fn pure_insertion() {
        let v = diff("ac", "abc", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().any(|(op, c)| *op == Op::Insert && *c == 'b'));
        }
    }

    #[test]
    fn pure_deletion() {
        let v = diff("abc", "ac", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().any(|(op, c)| *op == Op::Delete && *c == 'b'));
        }
    }

    #[test]
    fn empty_both_rejected() {
        assert_eq!(diff("", "", 100), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn empty_a_all_inserts() {
        let v = diff("", "abc", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().all(|(op, _)| *op == Op::Insert));
        }
    }

    #[test]
    fn empty_b_all_deletes() {
        let v = diff("abc", "", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().all(|(op, _)| *op == Op::Delete));
        }
    }

    #[test]
    fn over_max_len_rejected() {
        let big = "x".repeat(200);
        assert_eq!(diff(&big, "y", 100), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = diff("hello", "hallo", 100);
        let r2 = diff("hello", "hallo", 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_diff_works() {
        let v = diff("café", "cafe", 100);
        if let DiffVerdict::Ok { ops } = v {
            assert!(ops.iter().any(|(op, _)| *op == Op::Delete));
            assert!(ops.iter().any(|(op, _)| *op == Op::Insert));
        }
    }

    #[test]
    fn keeps_unchanged_prefix() {
        let v = diff("hello", "help", 100);
        if let DiffVerdict::Ok { ops } = v {
            // "hel" should be Same.
            let same_count = ops.iter().filter(|(op, _)| *op == Op::Same).count();
            assert!(same_count >= 3);
        }
    }

    #[test]
    fn complete_replacement() {
        let v = diff("abc", "xyz", 100);
        if let DiffVerdict::Ok { ops } = v {
            let same_count = ops.iter().filter(|(op, _)| *op == Op::Same).count();
            assert_eq!(same_count, 0);
        }
    }
}
