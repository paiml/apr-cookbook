//! # TUI Undo/Redo Stack
//!
//! Manage bounded undo/redo stack: pushing a new action clears the
//! redo stack. Returns final undo/redo depths after a sequence of
//! operations.
//!
//! Demonstrates the **TUI.102** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: command pattern (Gamma et al., GoF 1994); vim u/Ctrl-r
//!  undo semantics.
//!
//! Run with: cargo run --example tui_undo_redo_stack
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Action {
    Push,
    Undo,
    Redo,
}

#[derive(Debug, PartialEq)]
pub enum StackVerdict {
    Ok { undo_depth: u32, redo_depth: u32 },
    InvalidConfig,
}

pub fn execute(actions: &[Action], max_depth: u32) -> StackVerdict {
    if actions.is_empty() || max_depth == 0 {
        return StackVerdict::InvalidConfig;
    }
    let mut undo: Vec<u32> = Vec::new();
    let mut redo: Vec<u32> = Vec::new();
    let mut counter = 0u32;
    for action in actions {
        match action {
            Action::Push => {
                redo.clear();
                if undo.len() as u32 >= max_depth {
                    undo.remove(0);
                }
                counter += 1;
                undo.push(counter);
            }
            Action::Undo => {
                if let Some(top) = undo.pop() {
                    redo.push(top);
                }
            }
            Action::Redo => {
                if let Some(top) = redo.pop() {
                    undo.push(top);
                }
            }
        }
    }
    StackVerdict::Ok {
        undo_depth: undo.len() as u32,
        redo_depth: redo.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_undo_redo_stack")?;

    println!(
        "push x3, undo: {:?}",
        execute(
            &[Action::Push, Action::Push, Action::Push, Action::Undo],
            10
        )
    );
    println!(
        "push undo redo: {:?}",
        execute(&[Action::Push, Action::Undo, Action::Redo], 10)
    );
    println!("invalid: {:?}", execute(&[], 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn push_increases_undo() {
        let v = execute(&[Action::Push, Action::Push], 10);
        if let StackVerdict::Ok { undo_depth, .. } = v {
            assert_eq!(undo_depth, 2);
        }
    }

    #[test]
    fn undo_moves_to_redo() {
        let v = execute(&[Action::Push, Action::Undo], 10);
        if let StackVerdict::Ok {
            undo_depth,
            redo_depth,
        } = v
        {
            assert_eq!(undo_depth, 0);
            assert_eq!(redo_depth, 1);
        }
    }

    #[test]
    fn redo_returns_to_undo() {
        let v = execute(&[Action::Push, Action::Undo, Action::Redo], 10);
        if let StackVerdict::Ok {
            undo_depth,
            redo_depth,
        } = v
        {
            assert_eq!(undo_depth, 1);
            assert_eq!(redo_depth, 0);
        }
    }

    #[test]
    fn new_push_clears_redo() {
        let v = execute(&[Action::Push, Action::Undo, Action::Push], 10);
        if let StackVerdict::Ok { redo_depth, .. } = v {
            assert_eq!(redo_depth, 0);
        }
    }

    #[test]
    fn empty_actions_rejected() {
        assert_eq!(execute(&[], 10), StackVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_depth_rejected() {
        assert_eq!(execute(&[Action::Push], 0), StackVerdict::InvalidConfig);
    }

    #[test]
    fn max_depth_caps_undo() {
        let actions: Vec<Action> = vec![Action::Push; 10];
        let v = execute(&actions, 3);
        if let StackVerdict::Ok { undo_depth, .. } = v {
            assert_eq!(undo_depth, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = execute(&[Action::Push], 10);
        let r2 = execute(&[Action::Push], 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn undo_at_empty_no_op() {
        let v = execute(&[Action::Undo], 10);
        if let StackVerdict::Ok {
            undo_depth,
            redo_depth,
        } = v
        {
            assert_eq!(undo_depth, 0);
            assert_eq!(redo_depth, 0);
        }
    }

    #[test]
    fn redo_at_empty_no_op() {
        let v = execute(&[Action::Redo], 10);
        if let StackVerdict::Ok {
            undo_depth,
            redo_depth,
        } = v
        {
            assert_eq!(undo_depth, 0);
            assert_eq!(redo_depth, 0);
        }
    }

    #[test]
    fn many_undos_no_underflow() {
        let actions: Vec<Action> = vec![Action::Undo; 10];
        let v = execute(&actions, 10);
        if let StackVerdict::Ok { undo_depth, .. } = v {
            assert_eq!(undo_depth, 0);
        }
    }
}
