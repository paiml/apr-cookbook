//! # apr diff --structural — DiffSummary Renderer
//!
//! `apr diff --structural <A> <B>` summarises adds, removes, and
//! changed tensors. Output line format: `{symbol} {name}` where
//! symbol ∈ {`+`, `-`, `~`}. This recipe builds the renderer with
//! sorted output for diff-friendly CI logs.
//!
//! Demonstrates the **DIFF.4** recipe for PMAT-118 (apr diff coverage —
//! closing F-invariant gap from 1 → 4 recipes).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIFF-001 + Unix diff(1) conventions
//!
//! Run with: cargo run --example cli_diff_structural_renderer
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffOp {
    Added,
    Removed,
    Changed,
}

impl DiffOp {
    pub fn symbol(self) -> char {
        match self {
            DiffOp::Added => '+',
            DiffOp::Removed => '-',
            DiffOp::Changed => '~',
        }
    }
}

pub fn render_diff(left_names: &[&str], right_names: &[&str], changed: &[&str]) -> String {
    let lset: HashSet<&str> = left_names.iter().copied().collect();
    let rset: HashSet<&str> = right_names.iter().copied().collect();
    let cset: HashSet<&str> = changed.iter().copied().collect();

    let mut entries: Vec<(DiffOp, &str)> = Vec::new();
    let mut all: Vec<&str> = lset.union(&rset).copied().collect();
    all.sort_unstable();
    for name in all {
        if cset.contains(name) && lset.contains(name) && rset.contains(name) {
            entries.push((DiffOp::Changed, name));
        } else if !lset.contains(name) {
            entries.push((DiffOp::Added, name));
        } else if !rset.contains(name) {
            entries.push((DiffOp::Removed, name));
        }
    }
    entries
        .iter()
        .map(|(op, n)| format!("{} {}", op.symbol(), n))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn count_ops(
    left_names: &[&str],
    right_names: &[&str],
    changed: &[&str],
) -> (usize, usize, usize) {
    let lset: HashSet<&str> = left_names.iter().copied().collect();
    let rset: HashSet<&str> = right_names.iter().copied().collect();
    let cset: HashSet<&str> = changed.iter().copied().collect();
    let added = rset.difference(&lset).count();
    let removed = lset.difference(&rset).count();
    let changed_count = cset
        .iter()
        .filter(|n| lset.contains(*n) && rset.contains(*n))
        .count();
    (added, removed, changed_count)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diff_structural_renderer")?;

    let l = ["embed.weight", "layer.0", "layer.1", "removed_proj"];
    let r = ["embed.weight", "layer.0", "layer.1", "added_proj"];
    let changed = ["layer.0"];
    println!("{}", render_diff(&l, &r, &changed));
    println!("counts: {:?}", count_ops(&l, &r, &changed));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn added_uses_plus_symbol() {
        let r = render_diff(&[], &["a"], &[]);
        assert!(r.contains("+ a"));
    }

    #[test]
    fn removed_uses_minus_symbol() {
        let r = render_diff(&["a"], &[], &[]);
        assert!(r.contains("- a"));
    }

    #[test]
    fn changed_uses_tilde_symbol() {
        let r = render_diff(&["a"], &["a"], &["a"]);
        assert!(r.contains("~ a"));
    }

    #[test]
    fn output_sorted_alphabetically() {
        let r = render_diff(&[], &["z", "a", "m"], &[]);
        let lines: Vec<&str> = r.lines().collect();
        assert_eq!(lines, vec!["+ a", "+ m", "+ z"]);
    }

    #[test]
    fn unchanged_names_not_in_output() {
        // No diff ops on names present in both and not changed.
        let r = render_diff(&["a", "b"], &["a", "b"], &[]);
        assert_eq!(r, "");
    }

    #[test]
    fn changed_only_when_in_both() {
        // "removed" is in changed but only in left → emits `-`, not `~`.
        let r = render_diff(&["removed"], &[], &["removed"]);
        assert!(r.contains("- removed"));
        assert!(!r.contains("~ removed"));
    }

    #[test]
    fn count_ops_returns_3_tuple() {
        let l = ["a", "b", "c"];
        let r = ["b", "c", "d"];
        let c = ["b"];
        // added: d (1); removed: a (1); changed: b (1).
        assert_eq!(count_ops(&l, &r, &c), (1, 1, 1));
    }

    #[test]
    fn empty_diff_zero_counts() {
        assert_eq!(count_ops(&[], &[], &[]), (0, 0, 0));
    }

    #[test]
    fn count_ops_no_overlap_all_added_or_removed() {
        let l = ["a"];
        let r = ["b"];
        assert_eq!(count_ops(&l, &r, &[]), (1, 1, 0));
    }
}
