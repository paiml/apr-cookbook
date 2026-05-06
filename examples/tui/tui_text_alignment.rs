//! # TUI Text Alignment
//!
//! Align a single line of text within a fixed `width`. Supports
//! Left, Center, Right (justify deferred — most TUIs use single
//! word break). Returns the padded string.
//!
//! Demonstrates the **TUI.71** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS text-align spec; ANSI ESC[m display attributes.
//!
//! Run with: cargo run --example tui_text_alignment
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Align {
    Left,
    Center,
    Right,
}

#[derive(Debug, PartialEq)]
pub enum AlignVerdict {
    Ok { rendered: String, truncated: bool },
    InvalidConfig,
}

pub fn align(text: &str, width: u32, mode: Align) -> AlignVerdict {
    if width == 0 {
        return AlignVerdict::InvalidConfig;
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len() as u32;
    if len > width {
        // Truncate to width.
        let truncated: String = chars.iter().take(width as usize).collect();
        return AlignVerdict::Ok {
            rendered: truncated,
            truncated: true,
        };
    }
    let pad = width - len;
    let rendered = match mode {
        Align::Left => format!("{text}{}", " ".repeat(pad as usize)),
        Align::Right => format!("{}{text}", " ".repeat(pad as usize)),
        Align::Center => {
            let left = pad / 2;
            let right = pad - left;
            format!(
                "{}{text}{}",
                " ".repeat(left as usize),
                " ".repeat(right as usize)
            )
        }
    };
    AlignVerdict::Ok {
        rendered,
        truncated: false,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_text_alignment")?;

    println!("left: {:?}", align("hi", 10, Align::Left));
    println!("center: {:?}", align("hi", 10, Align::Center));
    println!("right: {:?}", align("hi", 10, Align::Right));
    println!("truncated: {:?}", align("very long", 4, Align::Left));
    println!("invalid: {:?}", align("x", 0, Align::Left));
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
        let v = align("hi", 5, Align::Left);
        if let AlignVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "hi   ");
        }
    }

    #[test]
    fn right_align_pads_left() {
        let v = align("hi", 5, Align::Right);
        if let AlignVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "   hi");
        }
    }

    #[test]
    fn center_align_pads_both_sides() {
        let v = align("hi", 6, Align::Center);
        if let AlignVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "  hi  ");
        }
    }

    #[test]
    fn center_align_uneven_padding_left_lighter() {
        let v = align("hi", 5, Align::Center);
        if let AlignVerdict::Ok { rendered, .. } = v {
            // pad=3 → left=1, right=2.
            assert_eq!(rendered, " hi  ");
        }
    }

    #[test]
    fn truncates_when_too_long() {
        let v = align("hello world", 5, Align::Left);
        if let AlignVerdict::Ok {
            rendered,
            truncated,
        } = v
        {
            assert_eq!(rendered.chars().count(), 5);
            assert!(truncated);
        }
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(align("x", 0, Align::Left), AlignVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = align("hello", 10, Align::Center);
        let r2 = align("hello", 10, Align::Center);
        assert_eq!(r1, r2);
    }

    #[test]
    fn exact_fit_no_padding() {
        let v = align("hello", 5, Align::Left);
        if let AlignVerdict::Ok {
            rendered,
            truncated,
        } = v
        {
            assert_eq!(rendered, "hello");
            assert!(!truncated);
        }
    }

    #[test]
    fn unicode_text_handled_by_char_count() {
        let v = align("café", 6, Align::Left);
        if let AlignVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.chars().count(), 6);
        }
    }

    #[test]
    fn empty_text_only_padding() {
        let v = align("", 4, Align::Center);
        if let AlignVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "    ");
        }
    }

    #[test]
    fn truncated_flag_false_when_fits() {
        let v = align("hi", 5, Align::Center);
        if let AlignVerdict::Ok { truncated, .. } = v {
            assert!(!truncated);
        }
    }
}
