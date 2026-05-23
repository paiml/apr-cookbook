//! # TUI Cursor Shape Phase
//!
//! Determine cursor shape (Block/Bar/Underline) given mode (insert/
//! normal/visual). Returns the shape and the matching VT100 escape
//! sequence ID.
//!
//! Demonstrates the **TUI.173** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VT520 cursor-style escape `CSI Ps SP q`; vim
//!  `set guicursor` mode-specific shape.
//!
//! Run with: cargo run --example tui_cursor_shape_phase
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CursorShape {
    Block,
    Bar,
    Underline,
}

#[derive(Debug, PartialEq)]
pub enum ShapeVerdict {
    Ok { shape: CursorShape, vt_code: u32 },
    InvalidConfig,
}

pub fn shape_for_mode(mode: &str) -> ShapeVerdict {
    match mode {
        "normal" => ShapeVerdict::Ok {
            shape: CursorShape::Block,
            vt_code: 2,
        },
        "insert" => ShapeVerdict::Ok {
            shape: CursorShape::Bar,
            vt_code: 6,
        },
        "visual" | "select" => ShapeVerdict::Ok {
            shape: CursorShape::Underline,
            vt_code: 4,
        },
        _ => ShapeVerdict::InvalidConfig,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_cursor_shape_phase")?;

    println!("normal: {:?}", shape_for_mode("normal"));
    println!("insert: {:?}", shape_for_mode("insert"));
    println!("visual: {:?}", shape_for_mode("visual"));
    println!("invalid: {:?}", shape_for_mode("xyz"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_mode_block() {
        let v = shape_for_mode("normal");
        if let ShapeVerdict::Ok { shape, vt_code } = v {
            assert_eq!(shape, CursorShape::Block);
            assert_eq!(vt_code, 2);
        }
    }

    #[test]
    fn insert_mode_bar() {
        let v = shape_for_mode("insert");
        if let ShapeVerdict::Ok { shape, vt_code } = v {
            assert_eq!(shape, CursorShape::Bar);
            assert_eq!(vt_code, 6);
        }
    }

    #[test]
    fn visual_mode_underline() {
        let v = shape_for_mode("visual");
        if let ShapeVerdict::Ok { shape, vt_code } = v {
            assert_eq!(shape, CursorShape::Underline);
            assert_eq!(vt_code, 4);
        }
    }

    #[test]
    fn select_mode_underline() {
        let v = shape_for_mode("select");
        if let ShapeVerdict::Ok { shape, .. } = v {
            assert_eq!(shape, CursorShape::Underline);
        }
    }

    #[test]
    fn unknown_mode_invalid() {
        assert_eq!(shape_for_mode("xyz"), ShapeVerdict::InvalidConfig);
    }

    #[test]
    fn empty_mode_invalid() {
        assert_eq!(shape_for_mode(""), ShapeVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = shape_for_mode("normal");
        let r2 = shape_for_mode("normal");
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        // "Normal" (capitalized) is unknown.
        assert_eq!(shape_for_mode("Normal"), ShapeVerdict::InvalidConfig);
    }

    #[test]
    fn vt_codes_are_unique_per_shape() {
        let n = shape_for_mode("normal");
        let i = shape_for_mode("insert");
        let v = shape_for_mode("visual");
        if let (
            ShapeVerdict::Ok { vt_code: c_n, .. },
            ShapeVerdict::Ok { vt_code: c_i, .. },
            ShapeVerdict::Ok { vt_code: c_v, .. },
        ) = (n, i, v)
        {
            assert_ne!(c_n, c_i);
            assert_ne!(c_i, c_v);
            assert_ne!(c_n, c_v);
        }
    }

    #[test]
    fn vt_code_in_valid_range() {
        let v = shape_for_mode("normal");
        if let ShapeVerdict::Ok { vt_code, .. } = v {
            // VT520 cursor style codes are 0..=6.
            assert!(vt_code <= 6);
        }
    }

    #[test]
    fn unicode_mode_invalid() {
        assert_eq!(shape_for_mode("café"), ShapeVerdict::InvalidConfig);
    }

    #[test]
    fn whitespace_mode_invalid() {
        assert_eq!(shape_for_mode("   "), ShapeVerdict::InvalidConfig);
    }
}
