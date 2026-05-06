//! # TUI Message Box Text Alignment
//!
//! Center, left-, or right-align text within a fixed-width box.
//! Returns the padded line; truncates with ellipsis if too long.
//!
//! Demonstrates the **TUI.36** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ANSI text-alignment + Unicode word boundaries.
//!
//! Run with: cargo run --example tui_message_box_align
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

#[derive(Debug, PartialEq)]
pub enum AlignVerdict {
    Ok { padded: String, truncated: bool },
    InvalidWidth,
}

pub fn align(text: &str, width: usize, alignment: Alignment) -> AlignVerdict {
    if width == 0 {
        return AlignVerdict::InvalidWidth;
    }
    let n = text.chars().count();
    if n > width {
        let kept: String = text.chars().take(width.saturating_sub(1)).collect();
        return AlignVerdict::Ok {
            padded: format!("{kept}…"),
            truncated: true,
        };
    }
    let pad = width - n;
    let padded = match alignment {
        Alignment::Left => format!("{text}{}", " ".repeat(pad)),
        Alignment::Right => format!("{}{text}", " ".repeat(pad)),
        Alignment::Center => {
            let l = pad / 2;
            let r = pad - l;
            format!("{}{text}{}", " ".repeat(l), " ".repeat(r))
        }
    };
    AlignVerdict::Ok {
        padded,
        truncated: false,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_message_box_align")?;

    println!("center: {:?}", align("hello", 10, Alignment::Center));
    println!("left: {:?}", align("hello", 10, Alignment::Left));
    println!("right: {:?}", align("hello", 10, Alignment::Right));
    println!("truncate: {:?}", align("hello world", 5, Alignment::Center));
    println!("invalid: {:?}", align("hello", 0, Alignment::Center));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn left_align_pads_right() {
        let v = align("abc", 6, Alignment::Left);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded, "abc   ");
        }
    }

    #[test]
    fn right_align_pads_left() {
        let v = align("abc", 6, Alignment::Right);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded, "   abc");
        }
    }

    #[test]
    fn center_align_balances() {
        let v = align("abc", 7, Alignment::Center);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded, "  abc  ");
        }
    }

    #[test]
    fn center_uneven_pad_extra_right() {
        let v = align("abc", 6, Alignment::Center);
        if let AlignVerdict::Ok { padded, .. } = v {
            // 1 left, 2 right.
            assert_eq!(padded, " abc  ");
        }
    }

    #[test]
    fn over_width_truncated() {
        let v = align("hello world", 5, Alignment::Center);
        if let AlignVerdict::Ok { truncated, padded } = v {
            assert!(truncated);
            assert!(padded.ends_with('…'));
        }
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(
            align("abc", 0, Alignment::Center),
            AlignVerdict::InvalidWidth
        );
    }

    #[test]
    fn empty_text_padded() {
        let v = align("", 5, Alignment::Center);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded, "     ");
        }
    }

    #[test]
    fn exact_fit_no_padding() {
        let v = align("hello", 5, Alignment::Center);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded, "hello");
        }
    }

    #[test]
    fn unicode_text_works() {
        let v = align("héllo", 7, Alignment::Center);
        if let AlignVerdict::Ok { padded, .. } = v {
            assert_eq!(padded.chars().count(), 7);
        }
    }

    #[test]
    fn deterministic() {
        let a = align("hi", 5, Alignment::Center);
        let b = align("hi", 5, Alignment::Center);
        assert_eq!(a, b);
    }
}
