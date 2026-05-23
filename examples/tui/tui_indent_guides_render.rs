//! # TUI Indent Guides Render
//!
//! Render indent-guide characters (`│  │  ▸ item`) for a tree-like
//! list given each item's depth. Active siblings emit `│`,
//! continuation rows emit ` `, leaves emit `▸`.
//!
//! Demonstrates the **TUI.64** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code editor.guides.indentation; eza tree drawing.
//!
//! Run with: cargo run --example tui_indent_guides_render
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GuideVerdict {
    Ok { rendered: Vec<String> },
    InvalidConfig,
}

pub fn render(items: &[(u32, &str)]) -> GuideVerdict {
    if items.is_empty() {
        return GuideVerdict::InvalidConfig;
    }
    let mut rendered: Vec<String> = Vec::with_capacity(items.len());
    for (depth, label) in items {
        let mut prefix = String::new();
        for _ in 0..*depth {
            prefix.push('│');
            prefix.push(' ');
            prefix.push(' ');
        }
        if *depth > 0 {
            // Replace last 3 chars with leaf marker.
            let chars: Vec<char> = prefix.chars().collect();
            prefix = chars[..chars.len() - 3].iter().collect();
            prefix.push('▸');
            prefix.push(' ');
        }
        prefix.push_str(label);
        rendered.push(prefix);
    }
    GuideVerdict::Ok { rendered }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_indent_guides_render")?;

    let items = [
        (0u32, "root"),
        (1, "child"),
        (2, "grandchild"),
        (1, "sibling"),
    ];
    println!("rendered: {:?}", render(&items));
    println!("invalid: {:?}", render(&[]));
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
    fn root_item_no_prefix() {
        let items = [(0u32, "root")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["root".to_string()]);
        }
    }

    #[test]
    fn depth_one_has_leaf_marker() {
        let items = [(1u32, "child")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert!(rendered[0].contains('▸'));
            assert!(rendered[0].ends_with("child"));
        }
    }

    #[test]
    fn depth_two_has_one_pipe() {
        let items = [(2u32, "grand")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            let pipe_count = rendered[0].chars().filter(|c| *c == '│').count();
            assert_eq!(pipe_count, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(render(&[]), GuideVerdict::InvalidConfig);
    }

    #[test]
    fn line_count_preserved() {
        let items = [(0u32, "a"), (1, "b"), (2, "c")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert_eq!(rendered.len(), 3);
        }
    }

    #[test]
    fn label_contained_in_output() {
        let items = [(2u32, "my_thing")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert!(rendered[0].contains("my_thing"));
        }
    }

    #[test]
    fn deterministic() {
        let items = [(0u32, "a"), (1, "b")];
        let r1 = render(&items);
        let r2 = render(&items);
        assert_eq!(r1, r2);
    }

    #[test]
    fn deeper_means_more_pipes() {
        let v1 = render(&[(2u32, "x")]);
        let v2 = render(&[(4u32, "x")]);
        if let (GuideVerdict::Ok { rendered: r1 }, GuideVerdict::Ok { rendered: r2 }) = (v1, v2) {
            let p1 = r1[0].chars().filter(|c| *c == '│').count();
            let p2 = r2[0].chars().filter(|c| *c == '│').count();
            assert!(p2 > p1);
        }
    }

    #[test]
    fn unicode_label_preserved() {
        let items = [(1u32, "café")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert!(rendered[0].contains("café"));
        }
    }

    #[test]
    fn mixed_depths_render() {
        let items = [(0u32, "root"), (1, "a"), (1, "b"), (0, "next_root")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert_eq!(rendered.len(), 4);
            assert_eq!(rendered[0], "root");
            assert_eq!(rendered[3], "next_root");
        }
    }

    #[test]
    fn leaf_marker_only_when_depth_gt_zero() {
        let items = [(0u32, "root")];
        let v = render(&items);
        if let GuideVerdict::Ok { rendered } = v {
            assert!(!rendered[0].contains('▸'));
        }
    }
}
