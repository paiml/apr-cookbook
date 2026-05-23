//! # TUI Select Popup Navigation
//!
//! Single-select popup state: index navigated via arrow keys, Enter
//! commits, Escape cancels. Returns the new state.
//!
//! Demonstrates the **TUI.31** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: dialog(1) menu / fzf select primitives.
//!
//! Run with: cargo run --example tui_select_popup
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectOp {
    Up,
    Down,
    Enter,
    Escape,
}

#[derive(Debug, PartialEq)]
pub enum SelectVerdict {
    Updated { index: u32 },
    Committed { index: u32 },
    Cancelled,
    InvalidConfig,
}

pub fn step(current_index: u32, item_count: u32, op: SelectOp) -> SelectVerdict {
    if item_count == 0 {
        return SelectVerdict::InvalidConfig;
    }
    let i = current_index.min(item_count - 1);
    match op {
        SelectOp::Up => SelectVerdict::Updated {
            index: i.saturating_sub(1),
        },
        SelectOp::Down => SelectVerdict::Updated {
            index: (i + 1).min(item_count - 1),
        },
        SelectOp::Enter => SelectVerdict::Committed { index: i },
        SelectOp::Escape => SelectVerdict::Cancelled,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_select_popup")?;

    println!("down: {:?}", step(0, 5, SelectOp::Down));
    println!("up at 0: {:?}", step(0, 5, SelectOp::Up));
    println!("commit: {:?}", step(2, 5, SelectOp::Enter));
    println!("cancel: {:?}", step(2, 5, SelectOp::Escape));
    println!("invalid: {:?}", step(0, 0, SelectOp::Down));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stepper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn down_advances() {
        let v = step(0, 5, SelectOp::Down);
        if let SelectVerdict::Updated { index } = v {
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn up_at_top_clamps() {
        let v = step(0, 5, SelectOp::Up);
        if let SelectVerdict::Updated { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn down_at_bottom_clamps() {
        let v = step(4, 5, SelectOp::Down);
        if let SelectVerdict::Updated { index } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn enter_commits_current() {
        let v = step(2, 5, SelectOp::Enter);
        if let SelectVerdict::Committed { index } = v {
            assert_eq!(index, 2);
        }
    }

    #[test]
    fn escape_cancels() {
        let v = step(2, 5, SelectOp::Escape);
        assert_eq!(v, SelectVerdict::Cancelled);
    }

    #[test]
    fn invalid_zero_items() {
        assert_eq!(step(0, 0, SelectOp::Down), SelectVerdict::InvalidConfig);
    }

    #[test]
    fn out_of_bounds_index_clamped() {
        let v = step(100, 5, SelectOp::Up);
        if let SelectVerdict::Updated { index } = v {
            assert_eq!(index, 3);
        }
    }

    #[test]
    fn commit_with_oob_clamps() {
        let v = step(100, 5, SelectOp::Enter);
        if let SelectVerdict::Committed { index } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn single_item_works() {
        let v = step(0, 1, SelectOp::Down);
        if let SelectVerdict::Updated { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn deterministic() {
        let a = step(2, 5, SelectOp::Down);
        let b = step(2, 5, SelectOp::Down);
        assert_eq!(a, b);
    }
}
