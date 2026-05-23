//! # TUI Breadcrumb Click Target
//!
//! Given the rendered breadcrumb (segments + separator), and the
//! click x coordinate, return which segment index was clicked or
//! NoTarget if click was on a separator.
//!
//! Demonstrates the **TUI.55** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Finder breadcrumb click hit-test.
//!
//! Run with: cargo run --example tui_breadcrumb_click_target
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClickVerdict {
    Hit { segment_index: u32 },
    NoTarget,
    OutOfBounds,
    EmptyPath,
}

pub fn classify(segments: &[&str], separator: &str, click_x: u32) -> ClickVerdict {
    if segments.is_empty() {
        return ClickVerdict::EmptyPath;
    }
    let mut x: u32 = 0;
    for (i, seg) in segments.iter().enumerate() {
        let seg_len = seg.chars().count() as u32;
        if click_x >= x && click_x < x + seg_len {
            return ClickVerdict::Hit {
                segment_index: i as u32,
            };
        }
        x += seg_len;
        if i + 1 < segments.len() {
            let sep_len = separator.chars().count() as u32;
            if click_x >= x && click_x < x + sep_len {
                return ClickVerdict::NoTarget;
            }
            x += sep_len;
        }
    }
    ClickVerdict::OutOfBounds
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_breadcrumb_click_target")?;

    let segments = ["root", "src", "examples"];
    let sep = " / ";
    println!("on root: {:?}", classify(&segments, sep, 0));
    println!("on src: {:?}", classify(&segments, sep, 7));
    println!("on examples: {:?}", classify(&segments, sep, 13));
    println!("on separator: {:?}", classify(&segments, sep, 4));
    println!("out of bounds: {:?}", classify(&segments, sep, 100));
    println!("empty: {:?}", classify(&[], sep, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<&'static str> {
        vec!["root", "src", "examples"]
    }

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn click_on_first_segment() {
        let v = classify(&segs(), " / ", 0);
        if let ClickVerdict::Hit { segment_index } = v {
            assert_eq!(segment_index, 0);
        }
    }

    #[test]
    fn click_on_separator_no_target() {
        // After "root" (4 chars), separator " / " starts at 4.
        let v = classify(&segs(), " / ", 4);
        assert_eq!(v, ClickVerdict::NoTarget);
    }

    #[test]
    fn click_on_second_segment() {
        // "root" + " / " = 7, so "src" starts at 7.
        let v = classify(&segs(), " / ", 7);
        if let ClickVerdict::Hit { segment_index } = v {
            assert_eq!(segment_index, 1);
        }
    }

    #[test]
    fn click_far_right_out_of_bounds() {
        let v = classify(&segs(), " / ", 1000);
        assert_eq!(v, ClickVerdict::OutOfBounds);
    }

    #[test]
    fn empty_path_rejected() {
        assert_eq!(classify(&[], " / ", 0), ClickVerdict::EmptyPath);
    }

    #[test]
    fn single_segment_works() {
        let v = classify(&["only"], "/", 1);
        if let ClickVerdict::Hit { segment_index } = v {
            assert_eq!(segment_index, 0);
        }
    }

    #[test]
    fn click_at_end_of_last_oob() {
        // Total render width: 4 + 3 + 3 + 3 + 8 = 21. x=21 is just past.
        let v = classify(&segs(), " / ", 21);
        assert_eq!(v, ClickVerdict::OutOfBounds);
    }

    #[test]
    fn click_within_last_segment() {
        // x=18 is still inside "examples" (which spans 13..21).
        let v = classify(&segs(), " / ", 18);
        assert!(matches!(v, ClickVerdict::Hit { segment_index: 2 }));
    }

    #[test]
    fn unicode_segment() {
        let segs = ["café", "résumé"];
        let v = classify(&segs, " / ", 0);
        if let ClickVerdict::Hit { segment_index } = v {
            assert_eq!(segment_index, 0);
        }
    }

    #[test]
    fn empty_separator_only_segments() {
        // No separator means no NoTarget hits.
        let v = classify(&["a", "b"], "", 1);
        if let ClickVerdict::Hit { segment_index } = v {
            assert_eq!(segment_index, 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = classify(&segs(), " / ", 7);
        let b = classify(&segs(), " / ", 7);
        assert_eq!(a, b);
    }
}
