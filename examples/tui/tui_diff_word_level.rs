//! # TUI Diff Word Level
//!
//! Diff two lines at word level: returns Same/Insert/Delete tagged
//! word sequence (whitespace-tokenized).
//!
//! Demonstrates the **TUI.128** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub PR word-diff view; git diff `--word-diff` mode.
//!
//! Run with: cargo run --example tui_diff_word_level
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DiffOp {
    Same,
    Insert,
    Delete,
}

#[derive(Debug, PartialEq)]
pub enum WordDiffVerdict {
    Ok {
        ops: Vec<(DiffOp, String)>,
        same_count: u32,
        change_count: u32,
    },
    InvalidConfig,
}

pub fn diff(a: &str, b: &str) -> WordDiffVerdict {
    if a.is_empty() && b.is_empty() {
        return WordDiffVerdict::InvalidConfig;
    }
    let av: Vec<&str> = a.split_whitespace().collect();
    let bv: Vec<&str> = b.split_whitespace().collect();
    let n = av.len();
    let m = bv.len();
    // LCS table.
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
    let mut ops: Vec<(DiffOp, String)> = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && av[i - 1] == bv[j - 1] {
            ops.push((DiffOp::Same, av[i - 1].to_string()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            ops.push((DiffOp::Insert, bv[j - 1].to_string()));
            j -= 1;
        } else {
            ops.push((DiffOp::Delete, av[i - 1].to_string()));
            i -= 1;
        }
    }
    ops.reverse();
    let same_count = ops.iter().filter(|(op, _)| *op == DiffOp::Same).count() as u32;
    let change_count = ops.len() as u32 - same_count;
    WordDiffVerdict::Ok {
        ops,
        same_count,
        change_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_diff_word_level")?;

    println!(
        "small change: {:?}",
        diff("the quick brown fox", "the slow brown fox")
    );
    println!("invalid: {:?}", diff("", ""));
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
        let v = diff("a b c", "a b c");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert!(ops.iter().all(|(op, _)| *op == DiffOp::Same));
        }
    }

    #[test]
    fn pure_insertion() {
        let v = diff("a c", "a b c");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert!(ops.iter().any(|(op, w)| *op == DiffOp::Insert && w == "b"));
        }
    }

    #[test]
    fn pure_deletion() {
        let v = diff("a b c", "a c");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert!(ops.iter().any(|(op, w)| *op == DiffOp::Delete && w == "b"));
        }
    }

    #[test]
    fn empty_both_rejected() {
        assert_eq!(diff("", ""), WordDiffVerdict::InvalidConfig);
    }

    #[test]
    fn empty_a_all_inserts() {
        let v = diff("", "a b");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert!(ops.iter().all(|(op, _)| *op == DiffOp::Insert));
        }
    }

    #[test]
    fn empty_b_all_deletes() {
        let v = diff("a b", "");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert!(ops.iter().all(|(op, _)| *op == DiffOp::Delete));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = diff("a b", "a c");
        let r2 = diff("a b", "a c");
        assert_eq!(r1, r2);
    }

    #[test]
    fn same_count_correct() {
        let v = diff("a b c", "a b d");
        if let WordDiffVerdict::Ok { same_count, .. } = v {
            assert_eq!(same_count, 2);
        }
    }

    #[test]
    fn change_count_correct() {
        let v = diff("a b c", "a x y");
        if let WordDiffVerdict::Ok { change_count, .. } = v {
            assert_eq!(change_count, 4);
        }
    }

    #[test]
    fn whitespace_tokenized() {
        let v = diff("hello   world", "hello world");
        if let WordDiffVerdict::Ok { ops, .. } = v {
            assert_eq!(ops.len(), 2);
        }
    }

    #[test]
    fn unicode_word_supported() {
        let v = diff("café", "café");
        if let WordDiffVerdict::Ok { same_count, .. } = v {
            assert_eq!(same_count, 1);
        }
    }
}
