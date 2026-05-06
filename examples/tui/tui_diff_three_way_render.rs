//! # TUI Three-Way Merge Conflict Render
//!
//! Render a Git-style three-way merge conflict block:
//!   `<<<<<<< HEAD` … `=======` … `>>>>>>> branch`.
//! Returns formatted lines.
//!
//! Demonstrates the **TUI.76** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Git merge.conflictstyle "merge"; RCS three-way merge.
//!
//! Run with: cargo run --example tui_diff_three_way_render
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MergeVerdict {
    Ok { lines: Vec<String> },
    InvalidConfig,
}

pub fn render(
    head_branch: &str,
    other_branch: &str,
    head_lines: &[&str],
    other_lines: &[&str],
) -> MergeVerdict {
    if head_branch.is_empty() || other_branch.is_empty() {
        return MergeVerdict::InvalidConfig;
    }
    if head_lines.is_empty() && other_lines.is_empty() {
        return MergeVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("<<<<<<< {head_branch}"));
    for l in head_lines {
        lines.push((*l).to_string());
    }
    lines.push("=======".to_string());
    for l in other_lines {
        lines.push((*l).to_string());
    }
    lines.push(format!(">>>>>>> {other_branch}"));
    MergeVerdict::Ok { lines }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_diff_three_way_render")?;

    let head = ["fn foo() -> i32 {", "    42", "}"];
    let other = ["fn foo() -> i32 {", "    99", "}"];
    println!("conflict: {:?}", render("HEAD", "feature/x", &head, &other));
    println!("invalid: {:?}", render("", "f", &head, &other));
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
    fn complete_conflict_structure() {
        let v = render("HEAD", "feat", &["a"], &["b"]);
        if let MergeVerdict::Ok { lines } = v {
            assert!(lines[0].starts_with("<<<<<<< HEAD"));
            assert_eq!(lines[1], "a");
            assert_eq!(lines[2], "=======");
            assert_eq!(lines[3], "b");
            assert!(lines[4].starts_with(">>>>>>> feat"));
        }
    }

    #[test]
    fn empty_head_branch_rejected() {
        assert_eq!(
            render("", "feat", &["a"], &["b"]),
            MergeVerdict::InvalidConfig
        );
    }

    #[test]
    fn empty_other_branch_rejected() {
        assert_eq!(
            render("HEAD", "", &["a"], &["b"]),
            MergeVerdict::InvalidConfig
        );
    }

    #[test]
    fn both_empty_rejected() {
        assert_eq!(
            render("HEAD", "feat", &[], &[]),
            MergeVerdict::InvalidConfig
        );
    }

    #[test]
    fn one_side_empty_works() {
        let v = render("HEAD", "feat", &["a"], &[]);
        if let MergeVerdict::Ok { lines } = v {
            assert_eq!(lines.len(), 4); // marker, a, separator, end
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("HEAD", "feat", &["x"], &["y"]);
        let r2 = render("HEAD", "feat", &["x"], &["y"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn line_count_matches_inputs() {
        let v = render("HEAD", "feat", &["a", "b"], &["c"]);
        if let MergeVerdict::Ok { lines } = v {
            // 1 marker + 2 head + 1 separator + 1 other + 1 end = 6.
            assert_eq!(lines.len(), 6);
        }
    }

    #[test]
    fn separator_always_seven_equals() {
        let v = render("HEAD", "feat", &["a"], &["b"]);
        if let MergeVerdict::Ok { lines } = v {
            assert_eq!(lines[2], "=======");
        }
    }

    #[test]
    fn marker_branches_in_correct_order() {
        let v = render("main", "develop", &["a"], &["b"]);
        if let MergeVerdict::Ok { lines } = v {
            assert!(lines[0].contains("main"));
            assert!(lines.last().unwrap().contains("develop"));
        }
    }

    #[test]
    fn marker_chars_correct() {
        let v = render("HEAD", "feat", &["a"], &["b"]);
        if let MergeVerdict::Ok { lines } = v {
            assert!(lines[0].starts_with("<<<<<<<"));
            assert!(lines.last().unwrap().starts_with(">>>>>>>"));
        }
    }

    #[test]
    fn unicode_branch_names_supported() {
        let v = render("café", "résumé", &["a"], &["b"]);
        if let MergeVerdict::Ok { lines } = v {
            assert!(lines[0].contains("café"));
            assert!(lines.last().unwrap().contains("résumé"));
        }
    }
}
