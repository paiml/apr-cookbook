//! # TUI Tag Chip Render
//!
//! Render a tag chip like `[bug]` or `[feature]`. If a tag exceeds
//! `max_width`, truncate with `…`. Returns rendered chips and any
//! truncated tag indices.
//!
//! Demonstrates the **TUI.75** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub label chip styling; Material Design Chips spec.
//!
//! Run with: cargo run --example tui_tag_chip_render
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChipVerdict {
    Ok {
        chips: Vec<String>,
        truncated_indices: Vec<u32>,
    },
    InvalidConfig,
}

pub fn render(tags: &[&str], max_width: u32) -> ChipVerdict {
    if tags.is_empty() || max_width < 3 {
        return ChipVerdict::InvalidConfig;
    }
    let mut chips: Vec<String> = Vec::with_capacity(tags.len());
    let mut truncated_indices: Vec<u32> = Vec::new();
    let inner_max = max_width - 2; // brackets cost 2.
    for (i, tag) in tags.iter().enumerate() {
        let len = tag.chars().count() as u32;
        let chip = if len <= inner_max {
            format!("[{tag}]")
        } else {
            truncated_indices.push(i as u32);
            let kept: String = tag.chars().take((inner_max - 1) as usize).collect();
            format!("[{kept}…]")
        };
        chips.push(chip);
    }
    ChipVerdict::Ok {
        chips,
        truncated_indices,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_tag_chip_render")?;

    let tags = ["bug", "feature", "performance-regression"];
    println!("render: {:?}", render(&tags, 12));
    println!("invalid: {:?}", render(&[], 10));
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
    fn short_tag_no_truncation() {
        let v = render(&["bug"], 10);
        if let ChipVerdict::Ok {
            chips,
            truncated_indices,
        } = v
        {
            assert_eq!(chips, vec!["[bug]".to_string()]);
            assert!(truncated_indices.is_empty());
        }
    }

    #[test]
    fn long_tag_truncated() {
        let v = render(&["performance-regression"], 8);
        if let ChipVerdict::Ok {
            chips,
            truncated_indices,
        } = v
        {
            assert!(chips[0].contains('…'));
            assert!(chips[0].chars().count() <= 8);
            assert_eq!(truncated_indices, vec![0]);
        }
    }

    #[test]
    fn empty_tags_rejected() {
        assert_eq!(render(&[], 10), ChipVerdict::InvalidConfig);
    }

    #[test]
    fn min_width_too_small_rejected() {
        // max_width < 3 leaves no room for content.
        assert_eq!(render(&["x"], 2), ChipVerdict::InvalidConfig);
    }

    #[test]
    fn count_matches_tag_count() {
        let v = render(&["a", "b", "c"], 10);
        if let ChipVerdict::Ok { chips, .. } = v {
            assert_eq!(chips.len(), 3);
        }
    }

    #[test]
    fn brackets_around_each_chip() {
        let v = render(&["x"], 10);
        if let ChipVerdict::Ok { chips, .. } = v {
            assert!(chips[0].starts_with('['));
            assert!(chips[0].ends_with(']'));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["bug", "feature"], 10);
        let r2 = render(&["bug", "feature"], 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_exact_max_no_trunc() {
        let v = render(&["abcdefgh"], 10);
        if let ChipVerdict::Ok {
            chips,
            truncated_indices,
        } = v
        {
            assert_eq!(chips[0].chars().count(), 10);
            assert!(truncated_indices.is_empty());
        }
    }

    #[test]
    fn one_over_max_truncated() {
        let v = render(&["abcdefghi"], 10);
        if let ChipVerdict::Ok {
            chips,
            truncated_indices,
        } = v
        {
            assert!(chips[0].contains('…'));
            assert_eq!(truncated_indices, vec![0]);
        }
    }

    #[test]
    fn unicode_tag_supported() {
        let v = render(&["café"], 10);
        if let ChipVerdict::Ok { chips, .. } = v {
            assert_eq!(chips[0], "[café]");
        }
    }

    #[test]
    fn mixed_truncation() {
        let v = render(&["short", "very_long_tag_name"], 8);
        if let ChipVerdict::Ok {
            truncated_indices, ..
        } = v
        {
            assert_eq!(truncated_indices, vec![1]);
        }
    }
}
