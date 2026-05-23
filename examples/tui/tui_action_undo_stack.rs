//! # TUI Action Undo Stack
//!
//! Maintain undo/redo state machine: push, undo, redo, clear redo
//! after a new push. Returns the current top-of-stack action.
//!
//! Demonstrates the **TUI.30** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: command pattern + Vim u/Ctrl-R semantics.
//!
//! Run with: cargo run --example tui_action_undo_stack
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoStack {
    pub undo: Vec<String>,
    pub redo: Vec<String>,
    pub max_depth: usize,
}

#[derive(Debug, PartialEq)]
pub enum StackVerdict {
    Ok {
        state: UndoStack,
        top: Option<String>,
    },
    NoUndo,
    NoRedo,
    InvalidDepth,
}

pub fn push(stack: &UndoStack, action: &str) -> StackVerdict {
    if stack.max_depth == 0 {
        return StackVerdict::InvalidDepth;
    }
    let mut new_undo = stack.undo.clone();
    new_undo.push(action.to_string());
    if new_undo.len() > stack.max_depth {
        new_undo.remove(0);
    }
    let new_state = UndoStack {
        undo: new_undo,
        redo: Vec::new(),
        max_depth: stack.max_depth,
    };
    let top = new_state.undo.last().cloned();
    StackVerdict::Ok {
        state: new_state,
        top,
    }
}

pub fn undo(stack: &UndoStack) -> StackVerdict {
    let mut new_undo = stack.undo.clone();
    let Some(action) = new_undo.pop() else {
        return StackVerdict::NoUndo;
    };
    let mut new_redo = stack.redo.clone();
    new_redo.push(action);
    let new_state = UndoStack {
        undo: new_undo,
        redo: new_redo,
        max_depth: stack.max_depth,
    };
    let top = new_state.undo.last().cloned();
    StackVerdict::Ok {
        state: new_state,
        top,
    }
}

pub fn redo(stack: &UndoStack) -> StackVerdict {
    let mut new_redo = stack.redo.clone();
    let Some(action) = new_redo.pop() else {
        return StackVerdict::NoRedo;
    };
    let mut new_undo = stack.undo.clone();
    new_undo.push(action);
    let new_state = UndoStack {
        undo: new_undo,
        redo: new_redo,
        max_depth: stack.max_depth,
    };
    let top = new_state.undo.last().cloned();
    StackVerdict::Ok {
        state: new_state,
        top,
    }
}

fn empty(max_depth: usize) -> UndoStack {
    UndoStack {
        undo: Vec::new(),
        redo: Vec::new(),
        max_depth,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_action_undo_stack")?;

    let s0 = empty(10);
    let s1 = if let StackVerdict::Ok { state, .. } = push(&s0, "type a") {
        state
    } else {
        s0
    };
    let s2 = if let StackVerdict::Ok { state, .. } = push(&s1, "type b") {
        state
    } else {
        s1
    };
    println!("after push: {:?}", s2);
    let s3 = if let StackVerdict::Ok { state, top } = undo(&s2) {
        println!("after undo, top: {:?}", top);
        state
    } else {
        s2
    };
    let _s4 = if let StackVerdict::Ok { state, top } = redo(&s3) {
        println!("after redo, top: {:?}", top);
        state
    } else {
        s3
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn push_adds_to_undo() {
        let s0 = empty(10);
        if let StackVerdict::Ok { state, top } = push(&s0, "a") {
            assert_eq!(state.undo, vec!["a".to_string()]);
            assert_eq!(top, Some("a".to_string()));
        }
    }

    #[test]
    fn push_clears_redo() {
        let mut s = empty(10);
        s.redo = vec!["old".to_string()];
        if let StackVerdict::Ok { state, .. } = push(&s, "new") {
            assert!(state.redo.is_empty());
        }
    }

    #[test]
    fn undo_moves_to_redo() {
        let mut s = empty(10);
        s.undo = vec!["a".to_string()];
        if let StackVerdict::Ok { state, .. } = undo(&s) {
            assert!(state.undo.is_empty());
            assert_eq!(state.redo, vec!["a".to_string()]);
        }
    }

    #[test]
    fn undo_empty_returns_no_undo() {
        let s = empty(10);
        assert_eq!(undo(&s), StackVerdict::NoUndo);
    }

    #[test]
    fn redo_pops_back() {
        let mut s = empty(10);
        s.redo = vec!["a".to_string()];
        if let StackVerdict::Ok { state, .. } = redo(&s) {
            assert_eq!(state.undo, vec!["a".to_string()]);
            assert!(state.redo.is_empty());
        }
    }

    #[test]
    fn redo_empty_returns_no_redo() {
        let s = empty(10);
        assert_eq!(redo(&s), StackVerdict::NoRedo);
    }

    #[test]
    fn max_depth_caps_undo() {
        let mut s = empty(2);
        if let StackVerdict::Ok { state, .. } = push(&s, "a") {
            s = state;
        }
        if let StackVerdict::Ok { state, .. } = push(&s, "b") {
            s = state;
        }
        if let StackVerdict::Ok { state, .. } = push(&s, "c") {
            assert_eq!(state.undo.len(), 2);
            assert_eq!(state.undo[0], "b");
        }
    }

    #[test]
    fn invalid_zero_depth() {
        let s = empty(0);
        assert_eq!(push(&s, "a"), StackVerdict::InvalidDepth);
    }

    #[test]
    fn undo_redo_round_trip() {
        let mut s = empty(10);
        if let StackVerdict::Ok { state, .. } = push(&s, "a") {
            s = state;
        }
        if let StackVerdict::Ok { state, .. } = undo(&s) {
            s = state;
        }
        if let StackVerdict::Ok { state, .. } = redo(&s) {
            assert_eq!(state.undo, vec!["a".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let s0 = empty(10);
        let a = push(&s0, "x");
        let b = push(&s0, "x");
        assert_eq!(a, b);
    }
}
