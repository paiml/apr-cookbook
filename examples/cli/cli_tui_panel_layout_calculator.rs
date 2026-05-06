//! # apr tui — Three-Panel Layout Calculator
//!
//! `apr tui <FILE>` shows a 3-panel layout: tensor list (left), preview
//! (center), metadata (right). Panel widths flex with terminal size.
//! This recipe builds the layout calculator and asserts the contract:
//! left fixed at 30 cols (or 25% of width, whichever is smaller), right
//! fixed at 40 cols (or 30% of width, whichever is smaller), center =
//! remaining. Below 80 cols total, panels stack vertically.
//!
//! Demonstrates the **TUI.4** recipe for PMAT-108 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-001 + ratatui flexbox conventions
//!
//! Run with: cargo run --example cli_tui_panel_layout_calculator
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Horizontal { left: u32, center: u32, right: u32 },
    Vertical { rows: u32 },
}

const MIN_HORIZONTAL_WIDTH: u32 = 80;
const LEFT_HARD_CAP: u32 = 30;
const RIGHT_HARD_CAP: u32 = 40;

pub fn compute_layout(width: u32, height: u32) -> LayoutMode {
    if width < MIN_HORIZONTAL_WIDTH {
        return LayoutMode::Vertical { rows: height };
    }
    let left = LEFT_HARD_CAP.min(width / 4);
    let right = RIGHT_HARD_CAP.min(3 * width / 10);
    let center = width.saturating_sub(left + right);
    LayoutMode::Horizontal {
        left,
        center,
        right,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_panel_layout_calculator")?;

    for (label, w, h) in [
        ("xterm 80×24", 80, 24),
        ("medium 120×40", 120, 40),
        ("wide 200×50", 200, 50),
        ("ultrawide 400×100", 400, 100),
        ("narrow 60×20", 60, 20),
    ] {
        println!("{label:>20} ({w}x{h})  →  {:?}", compute_layout(w, h));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn standard_80_col_horizontal() {
        let l = compute_layout(80, 24);
        assert!(matches!(l, LayoutMode::Horizontal { .. }));
    }

    #[test]
    fn below_80_cols_stacks_vertical() {
        assert!(matches!(
            compute_layout(60, 20),
            LayoutMode::Vertical { .. }
        ));
        assert!(matches!(
            compute_layout(40, 30),
            LayoutMode::Vertical { .. }
        ));
    }

    #[test]
    fn boundary_at_80_horizontal_not_vertical() {
        assert!(matches!(
            compute_layout(80, 24),
            LayoutMode::Horizontal { .. }
        ));
    }

    #[test]
    fn left_panel_capped_at_30_cols() {
        // wide layouts: left = min(30, width/4).
        // width 200 → width/4 = 50 → cap to 30.
        if let LayoutMode::Horizontal { left, .. } = compute_layout(200, 24) {
            assert_eq!(left, LEFT_HARD_CAP);
        } else {
            panic!("expected Horizontal");
        }
    }

    #[test]
    fn right_panel_capped_at_40_cols() {
        // wide layouts: right = min(40, 3*width/10).
        // width 200 → 3*200/10 = 60 → cap to 40.
        if let LayoutMode::Horizontal { right, .. } = compute_layout(200, 24) {
            assert_eq!(right, RIGHT_HARD_CAP);
        }
    }

    #[test]
    fn small_horizontal_uses_proportional_panels() {
        // width 80: left = min(30, 20) = 20; right = min(40, 24) = 24.
        if let LayoutMode::Horizontal { left, right, .. } = compute_layout(80, 24) {
            assert_eq!(left, 20);
            assert_eq!(right, 24);
        }
    }

    #[test]
    fn center_fills_remaining_width() {
        let w = 200;
        if let LayoutMode::Horizontal {
            left,
            center,
            right,
        } = compute_layout(w, 24)
        {
            assert_eq!(left + center + right, w);
        }
    }
}
