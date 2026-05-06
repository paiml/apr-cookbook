//! # TUI Marquee Scroll
//!
//! Compute the visible window of a long line that scrolls left
//! continuously. Returns the `width`-character substring at tick `t`,
//! wrapping around with a `gap` of spaces between repetitions.
//!
//! Demonstrates the **TUI.56** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML <marquee> behavior + ANSI text scroller.
//!
//! Run with: cargo run --example tui_marquee_scroll
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MarqueeVerdict {
    Ok { window: String, offset: u32 },
    InvalidConfig,
}

pub fn frame(text: &str, width: usize, gap: usize, tick: u64) -> MarqueeVerdict {
    if width == 0 || text.is_empty() {
        return MarqueeVerdict::InvalidConfig;
    }
    let n = text.chars().count();
    let cycle = n + gap;
    if cycle == 0 {
        return MarqueeVerdict::InvalidConfig;
    }
    let offset = (tick as usize) % cycle;
    let mut window = String::with_capacity(width);
    let extended: String = format!("{text}{}", " ".repeat(gap));
    let chars: Vec<char> = extended.chars().collect();
    for i in 0..width {
        let idx = (offset + i) % cycle;
        window.push(chars[idx]);
    }
    MarqueeVerdict::Ok {
        window,
        offset: offset as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_marquee_scroll")?;

    println!("tick 0: {:?}", frame("hello world", 8, 4, 0));
    println!("tick 5: {:?}", frame("hello world", 8, 4, 5));
    println!("wrap: {:?}", frame("hello world", 8, 4, 100));
    println!("invalid: {:?}", frame("hello", 0, 4, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn window_is_correct_width() {
        let v = frame("hello world", 8, 4, 0);
        if let MarqueeVerdict::Ok { window, .. } = v {
            assert_eq!(window.chars().count(), 8);
        }
    }

    #[test]
    fn tick_advances_offset() {
        let v0 = frame("hello world", 8, 4, 0);
        let v5 = frame("hello world", 8, 4, 5);
        if let (MarqueeVerdict::Ok { offset: o0, .. }, MarqueeVerdict::Ok { offset: o5, .. }) =
            (v0, v5)
        {
            assert_eq!(o5 - o0, 5);
        }
    }

    #[test]
    fn wraps_at_cycle() {
        let v = frame("hi", 4, 2, 0);
        let v_cycle = frame("hi", 4, 2, 4); // n+gap=4 → cycle.
        assert_eq!(v, v_cycle);
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(frame("hi", 0, 1, 0), MarqueeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_text() {
        assert_eq!(frame("", 8, 4, 0), MarqueeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_gap_works() {
        let v = frame("hello", 4, 0, 0);
        if let MarqueeVerdict::Ok { window, .. } = v {
            assert_eq!(window, "hell");
        }
    }

    #[test]
    fn unicode_text() {
        let v = frame("café", 4, 2, 0);
        if let MarqueeVerdict::Ok { window, .. } = v {
            assert!(window.contains('é'));
        }
    }

    #[test]
    fn very_large_tick_works() {
        let v = frame("hi", 4, 2, u64::MAX);
        assert!(matches!(v, MarqueeVerdict::Ok { .. }));
    }

    #[test]
    fn width_larger_than_text_pads() {
        let v = frame("abc", 8, 2, 0);
        if let MarqueeVerdict::Ok { window, .. } = v {
            assert_eq!(window.chars().count(), 8);
        }
    }

    #[test]
    fn deterministic() {
        let a = frame("hello", 4, 2, 5);
        let b = frame("hello", 4, 2, 5);
        assert_eq!(a, b);
    }
}
