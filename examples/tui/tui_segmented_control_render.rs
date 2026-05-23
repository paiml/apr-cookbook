//! # TUI Segmented Control Render
//!
//! Render a horizontal segmented control like
//! `[ All ][ Active ][ Done ]` with the selected segment highlighted.
//! Returns rendered string and per-segment click zones.
//!
//! Demonstrates the **TUI.88** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Cocoa NSSegmentedControl; iOS UISegmentedControl.
//!
//! Run with: cargo run --example tui_segmented_control_render
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SegmentVerdict {
    Ok {
        rendered: String,
        zones: Vec<(u32, u32)>,
    },
    InvalidConfig,
}

pub fn render(labels: &[&str], selected: u32) -> SegmentVerdict {
    if labels.is_empty() || (selected as usize) >= labels.len() {
        return SegmentVerdict::InvalidConfig;
    }
    let mut rendered = String::new();
    let mut zones: Vec<(u32, u32)> = Vec::with_capacity(labels.len());
    let mut cursor: u32 = 0;
    for (i, label) in labels.iter().enumerate() {
        let segment = if i as u32 == selected {
            format!("[*{label}*]")
        } else {
            format!("[ {label} ]")
        };
        let len = segment.chars().count() as u32;
        zones.push((cursor, cursor + len));
        rendered.push_str(&segment);
        cursor += len;
    }
    SegmentVerdict::Ok { rendered, zones }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_segmented_control_render")?;

    let labels = ["All", "Active", "Done"];
    println!("first selected: {:?}", render(&labels, 0));
    println!("middle selected: {:?}", render(&labels, 1));
    println!("invalid: {:?}", render(&[], 0));
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
    fn first_selected_marker() {
        let v = render(&["A", "B"], 0);
        if let SegmentVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with("[*A*]"));
        }
    }

    #[test]
    fn second_selected_no_first_marker() {
        let v = render(&["A", "B"], 1);
        if let SegmentVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("[ A ]"));
            assert!(rendered.contains("[*B*]"));
        }
    }

    #[test]
    fn empty_labels_rejected() {
        assert_eq!(render(&[], 0), SegmentVerdict::InvalidConfig);
    }

    #[test]
    fn out_of_range_rejected() {
        assert_eq!(render(&["A"], 5), SegmentVerdict::InvalidConfig);
    }

    #[test]
    fn zones_count_matches_labels() {
        let v = render(&["A", "B", "C"], 0);
        if let SegmentVerdict::Ok { zones, .. } = v {
            assert_eq!(zones.len(), 3);
        }
    }

    #[test]
    fn zones_non_overlapping() {
        let v = render(&["A", "B"], 0);
        if let SegmentVerdict::Ok { zones, .. } = v {
            assert_eq!(zones[0].1, zones[1].0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["A", "B"], 0);
        let r2 = render(&["A", "B"], 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_segment_works() {
        let v = render(&["only"], 0);
        if let SegmentVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "[*only*]");
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render(&["café"], 0);
        if let SegmentVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn zones_start_at_zero() {
        let v = render(&["A"], 0);
        if let SegmentVerdict::Ok { zones, .. } = v {
            assert_eq!(zones[0].0, 0);
        }
    }

    #[test]
    fn last_zone_end_equals_render_length() {
        let v = render(&["A", "B"], 0);
        if let SegmentVerdict::Ok { rendered, zones } = v {
            assert_eq!(zones.last().unwrap().1 as usize, rendered.chars().count());
        }
    }
}
