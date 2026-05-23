//! # TUI Fold Block Collapse
//!
//! Compute visible line set after collapsing fold ranges. Each fold
//! is `(start_line, end_line)` inclusive; collapsed folds hide
//! interior lines but keep the start as the fold marker.
//!
//! Demonstrates the **TUI.137** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim folding `:set foldmethod`; VS Code `editor.folding`
//!  expansion semantics.
//!
//! Run with: cargo run --example tui_fold_block_collapse
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum FoldVerdict {
    Ok {
        visible_lines: Vec<u32>,
        hidden_count: u32,
    },
    InvalidConfig,
}

pub fn collapse(total_lines: u32, folds: &[(u32, u32)]) -> FoldVerdict {
    if total_lines == 0 {
        return FoldVerdict::InvalidConfig;
    }
    for &(s, e) in folds {
        if s == 0 || e == 0 || s > e || e > total_lines {
            return FoldVerdict::InvalidConfig;
        }
    }
    let mut hidden: BTreeSet<u32> = BTreeSet::new();
    for &(s, e) in folds {
        // Hide interior of the fold (s+1..=e). Keep start as marker.
        for l in (s + 1)..=e {
            hidden.insert(l);
        }
    }
    let visible: Vec<u32> = (1..=total_lines).filter(|l| !hidden.contains(l)).collect();
    FoldVerdict::Ok {
        visible_lines: visible,
        hidden_count: hidden.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_fold_block_collapse")?;

    println!("fold (3-7): {:?}", collapse(10, &[(3, 7)]));
    println!("invalid: {:?}", collapse(0, &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_folds_all_visible() {
        let v = collapse(5, &[]);
        if let FoldVerdict::Ok { visible_lines, .. } = v {
            assert_eq!(visible_lines, vec![1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn single_fold_hides_interior() {
        let v = collapse(10, &[(3, 7)]);
        if let FoldVerdict::Ok { visible_lines, .. } = v {
            assert_eq!(visible_lines, vec![1, 2, 3, 8, 9, 10]);
        }
    }

    #[test]
    fn hidden_count_correct() {
        let v = collapse(10, &[(3, 7)]);
        if let FoldVerdict::Ok { hidden_count, .. } = v {
            // Lines 4,5,6,7 hidden = 4 lines.
            assert_eq!(hidden_count, 4);
        }
    }

    #[test]
    fn invalid_zero_lines() {
        assert_eq!(collapse(0, &[]), FoldVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_fold_zero_start() {
        assert_eq!(collapse(5, &[(0, 3)]), FoldVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_fold_inverted() {
        assert_eq!(collapse(5, &[(3, 1)]), FoldVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_fold_past_end() {
        assert_eq!(collapse(5, &[(3, 10)]), FoldVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = collapse(5, &[(2, 4)]);
        let r2 = collapse(5, &[(2, 4)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn multiple_folds_handled() {
        let v = collapse(15, &[(3, 5), (8, 10)]);
        if let FoldVerdict::Ok { visible_lines, .. } = v {
            // 1,2,3 (start), 6,7, 8 (start), 11,12,13,14,15 → 11 visible
            assert_eq!(visible_lines, vec![1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 15]);
        }
    }

    #[test]
    fn fold_single_line_no_hidden() {
        // (s,s) → only s remains, nothing hidden (empty interior).
        let v = collapse(5, &[(3, 3)]);
        if let FoldVerdict::Ok { hidden_count, .. } = v {
            assert_eq!(hidden_count, 0);
        }
    }

    #[test]
    fn overlapping_folds_handled() {
        let v = collapse(10, &[(2, 5), (4, 7)]);
        if let FoldVerdict::Ok { visible_lines, .. } = v {
            // hidden: 3,4,5 from first; 5,6,7 from second → 3,4,5,6,7
            // visible: 1,2 (start), 8,9,10
            // wait — second fold start=4 is hidden by first, so only 5,6,7
            // Actually deduped via BTreeSet: hidden = {3,4,5,6,7} → 1,2,8,9,10.
            assert_eq!(visible_lines, vec![1, 2, 8, 9, 10]);
        }
    }

    #[test]
    fn fold_at_end_handled() {
        let v = collapse(5, &[(3, 5)]);
        if let FoldVerdict::Ok { visible_lines, .. } = v {
            assert_eq!(visible_lines, vec![1, 2, 3]);
        }
    }
}
