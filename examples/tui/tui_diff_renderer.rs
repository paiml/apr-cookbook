//! # TUI Line-Diff Renderer
//!
//! Render line-level diff between old and new text. Produces a list of
//! (kind, line) entries: Added/Removed/Unchanged. Uses simple
//! longest-common-prefix matching (not full LCS).
//!
//! Demonstrates the **TUI.11** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU diff line mode + git's --stat output.
//!
//! Run with: cargo run --example tui_diff_renderer
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    Added,
    Removed,
    Unchanged,
}

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        entries: Vec<(LineKind, String)>,
        added_count: u32,
        removed_count: u32,
    },
    BothEmpty,
}

pub fn render(old_text: &str, new_text: &str) -> DiffVerdict {
    let old_lines: Vec<&str> = old_text.lines().collect();
    let new_lines: Vec<&str> = new_text.lines().collect();
    if old_lines.is_empty() && new_lines.is_empty() {
        return DiffVerdict::BothEmpty;
    }
    let mut entries: Vec<(LineKind, String)> = Vec::new();
    let mut added = 0u32;
    let mut removed = 0u32;
    let common_prefix = old_lines
        .iter()
        .zip(new_lines.iter())
        .take_while(|(a, b)| a == b)
        .count();
    for line in &old_lines[..common_prefix] {
        entries.push((LineKind::Unchanged, (*line).to_string()));
    }
    for line in &old_lines[common_prefix..] {
        entries.push((LineKind::Removed, (*line).to_string()));
        removed += 1;
    }
    for line in &new_lines[common_prefix..] {
        entries.push((LineKind::Added, (*line).to_string()));
        added += 1;
    }
    DiffVerdict::Ok {
        entries,
        added_count: added,
        removed_count: removed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_diff_renderer")?;

    println!("added: {:?}", render("a\nb\n", "a\nb\nc\n"));
    println!("removed: {:?}", render("a\nb\nc\n", "a\nb\n"));
    println!("identical: {:?}", render("a\nb\n", "a\nb\n"));
    println!("both empty: {:?}", render("", ""));
    println!("complete swap: {:?}", render("a\nb\n", "c\nd\n"));
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
    fn additions_at_end() {
        let v = render("a\nb\n", "a\nb\nc\n");
        if let DiffVerdict::Ok {
            added_count,
            removed_count,
            ..
        } = v
        {
            assert_eq!(added_count, 1);
            assert_eq!(removed_count, 0);
        }
    }

    #[test]
    fn removals_at_end() {
        let v = render("a\nb\nc\n", "a\nb\n");
        if let DiffVerdict::Ok {
            added_count,
            removed_count,
            ..
        } = v
        {
            assert_eq!(added_count, 0);
            assert_eq!(removed_count, 1);
        }
    }

    #[test]
    fn identical_no_changes() {
        let v = render("a\nb\n", "a\nb\n");
        if let DiffVerdict::Ok {
            added_count,
            removed_count,
            ..
        } = v
        {
            assert_eq!(added_count, 0);
            assert_eq!(removed_count, 0);
        }
    }

    #[test]
    fn both_empty_special() {
        assert_eq!(render("", ""), DiffVerdict::BothEmpty);
    }

    #[test]
    fn complete_swap() {
        let v = render("a\nb\n", "c\nd\n");
        if let DiffVerdict::Ok {
            added_count,
            removed_count,
            ..
        } = v
        {
            assert_eq!(removed_count, 2);
            assert_eq!(added_count, 2);
        }
    }

    #[test]
    fn old_empty_all_added() {
        let v = render("", "x\ny\n");
        if let DiffVerdict::Ok { added_count, .. } = v {
            assert_eq!(added_count, 2);
        }
    }

    #[test]
    fn new_empty_all_removed() {
        let v = render("x\ny\n", "");
        if let DiffVerdict::Ok { removed_count, .. } = v {
            assert_eq!(removed_count, 2);
        }
    }

    #[test]
    fn entry_kinds_correct() {
        let v = render("a\nb\n", "a\nx\n");
        if let DiffVerdict::Ok { entries, .. } = v {
            // First "a" is unchanged; "b" removed; "x" added.
            assert_eq!(entries[0].0, LineKind::Unchanged);
            assert_eq!(entries[1].0, LineKind::Removed);
            assert_eq!(entries[2].0, LineKind::Added);
        }
    }

    #[test]
    fn entry_count_matches_total() {
        let v = render("a\nb\n", "a\nx\ny\n");
        if let DiffVerdict::Ok { entries, .. } = v {
            // 1 unchanged + 1 removed + 2 added = 4.
            assert_eq!(entries.len(), 4);
        }
    }

    #[test]
    fn deterministic() {
        let a = render("a\nb\n", "a\nb\nc\n");
        let b = render("a\nb\n", "a\nb\nc\n");
        assert_eq!(a, b);
    }
}
