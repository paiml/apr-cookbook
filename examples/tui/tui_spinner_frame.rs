//! # TUI Spinner Frame Picker
//!
//! Pick spinner frame at a given tick. Returns the glyph and the
//! frame index. Supports common spinner styles (Braille, ASCII).
//!
//! Demonstrates the **TUI.14** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: indicatif/cli-spinners glyph dictionary.
//!
//! Run with: cargo run --example tui_spinner_frame
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinnerStyle {
    Braille,
    Dots,
    Pipe,
}

#[derive(Debug, PartialEq)]
pub enum SpinnerVerdict {
    Pick { glyph: char, frame_index: u32 },
}

const BRAILLE: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const DOTS: &[char] = &['.', 'o', 'O', '@', '*'];
const PIPE: &[char] = &['|', '/', '-', '\\'];

pub fn pick(style: SpinnerStyle, tick: u64) -> SpinnerVerdict {
    let frames: &[char] = match style {
        SpinnerStyle::Braille => BRAILLE,
        SpinnerStyle::Dots => DOTS,
        SpinnerStyle::Pipe => PIPE,
    };
    let n = frames.len() as u64;
    let idx = (tick % n) as u32;
    SpinnerVerdict::Pick {
        glyph: frames[idx as usize],
        frame_index: idx,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_spinner_frame")?;

    println!("braille[0]: {:?}", pick(SpinnerStyle::Braille, 0));
    println!("braille[5]: {:?}", pick(SpinnerStyle::Braille, 5));
    println!("dots[3]: {:?}", pick(SpinnerStyle::Dots, 3));
    println!("pipe[0]: {:?}", pick(SpinnerStyle::Pipe, 0));
    println!("wrap: {:?}", pick(SpinnerStyle::Pipe, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn braille_first_frame() {
        let v = pick(SpinnerStyle::Braille, 0);
        if let SpinnerVerdict::Pick { glyph, frame_index } = v {
            assert_eq!(glyph, '⠋');
            assert_eq!(frame_index, 0);
        }
    }

    #[test]
    fn dots_wraps() {
        // Dots has 5 frames; tick 7 → frame 2.
        let v = pick(SpinnerStyle::Dots, 7);
        if let SpinnerVerdict::Pick { frame_index, .. } = v {
            assert_eq!(frame_index, 2);
        }
    }

    #[test]
    fn pipe_advances() {
        let v0 = pick(SpinnerStyle::Pipe, 0);
        let v1 = pick(SpinnerStyle::Pipe, 1);
        if let (SpinnerVerdict::Pick { glyph: g0, .. }, SpinnerVerdict::Pick { glyph: g1, .. }) =
            (v0, v1)
        {
            assert_ne!(g0, g1);
        }
    }

    #[test]
    fn frame_index_in_range() {
        for tick in [0, 1, 5, 10, 100, 9999] {
            let v = pick(SpinnerStyle::Braille, tick);
            if let SpinnerVerdict::Pick { frame_index, .. } = v {
                assert!(frame_index < 10);
            }
        }
    }

    #[test]
    fn dots_frame_count() {
        for tick in 0..10 {
            let v = pick(SpinnerStyle::Dots, tick);
            if let SpinnerVerdict::Pick { frame_index, .. } = v {
                assert!(frame_index < 5);
            }
        }
    }

    #[test]
    fn pipe_frame_count() {
        for tick in 0..10 {
            let v = pick(SpinnerStyle::Pipe, tick);
            if let SpinnerVerdict::Pick { frame_index, .. } = v {
                assert!(frame_index < 4);
            }
        }
    }

    #[test]
    fn modulo_wraps() {
        // tick = N * frame_count → back to frame 0.
        let v = pick(SpinnerStyle::Pipe, 8); // 8 % 4 = 0
        if let SpinnerVerdict::Pick { frame_index, .. } = v {
            assert_eq!(frame_index, 0);
        }
    }

    #[test]
    fn glyph_from_known_set() {
        let v = pick(SpinnerStyle::Pipe, 1);
        if let SpinnerVerdict::Pick { glyph, .. } = v {
            assert!(matches!(glyph, '|' | '/' | '-' | '\\'));
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(SpinnerStyle::Braille, 5);
        let b = pick(SpinnerStyle::Braille, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn very_large_tick_works() {
        let v = pick(SpinnerStyle::Braille, u64::MAX);
        // No panic; just compute.
        assert!(matches!(v, SpinnerVerdict::Pick { .. }));
    }
}
