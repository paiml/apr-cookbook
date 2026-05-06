//! # TUI Clipboard Buffer
//!
//! Internal cut/copy/paste buffer with a max history of N entries
//! (most-recent first). Returns the new buffer state and current head.
//!
//! Demonstrates the **TUI.44** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Emacs kill-ring + macOS clipboard ring patterns.
//!
//! Run with: cargo run --example tui_clipboard_buffer
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clipboard {
    pub entries: Vec<String>,
    pub max_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipOp {
    Cut(String),
    Copy(String),
    Paste,
}

#[derive(Debug, PartialEq)]
pub enum ClipVerdict {
    Ok {
        state: Clipboard,
        pasted: Option<String>,
    },
    Empty,
    InvalidConfig,
}

pub fn apply(state: &Clipboard, op: ClipOp) -> ClipVerdict {
    if state.max_size == 0 {
        return ClipVerdict::InvalidConfig;
    }
    match op {
        ClipOp::Cut(text) | ClipOp::Copy(text) => {
            let mut new_entries = state.entries.clone();
            new_entries.insert(0, text);
            while new_entries.len() > state.max_size {
                new_entries.pop();
            }
            ClipVerdict::Ok {
                state: Clipboard {
                    entries: new_entries,
                    max_size: state.max_size,
                },
                pasted: None,
            }
        }
        ClipOp::Paste => {
            let Some(head) = state.entries.first() else {
                return ClipVerdict::Empty;
            };
            ClipVerdict::Ok {
                state: state.clone(),
                pasted: Some(head.clone()),
            }
        }
    }
}

fn empty(max_size: usize) -> Clipboard {
    Clipboard {
        entries: Vec::new(),
        max_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_clipboard_buffer")?;

    let s0 = empty(3);
    let s1 = if let ClipVerdict::Ok { state, .. } = apply(&s0, ClipOp::Copy("hello".to_string())) {
        state
    } else {
        s0.clone()
    };
    let s2 = if let ClipVerdict::Ok { state, .. } = apply(&s1, ClipOp::Cut("world".to_string())) {
        state
    } else {
        s1.clone()
    };
    println!("after cut: {s2:?}");
    if let ClipVerdict::Ok { pasted, .. } = apply(&s2, ClipOp::Paste) {
        println!("pasted: {pasted:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn copy_pushes_to_head() {
        let s = empty(5);
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("a".to_string())) {
            assert_eq!(state.entries, vec!["a".to_string()]);
        }
    }

    #[test]
    fn paste_returns_head() {
        let mut s = empty(5);
        s.entries = vec!["x".to_string()];
        if let ClipVerdict::Ok { pasted, .. } = apply(&s, ClipOp::Paste) {
            assert_eq!(pasted, Some("x".to_string()));
        }
    }

    #[test]
    fn paste_on_empty() {
        let s = empty(5);
        assert_eq!(apply(&s, ClipOp::Paste), ClipVerdict::Empty);
    }

    #[test]
    fn invalid_zero_max() {
        let s = empty(0);
        assert_eq!(
            apply(&s, ClipOp::Copy("x".to_string())),
            ClipVerdict::InvalidConfig
        );
    }

    #[test]
    fn capacity_drops_oldest() {
        let mut s = empty(2);
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("a".to_string())) {
            s = state;
        }
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("b".to_string())) {
            s = state;
        }
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("c".to_string())) {
            assert_eq!(state.entries, vec!["c".to_string(), "b".to_string()]);
        }
    }

    #[test]
    fn cut_treated_like_copy() {
        let s = empty(5);
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Cut("a".to_string())) {
            assert_eq!(state.entries, vec!["a".to_string()]);
        }
    }

    #[test]
    fn newer_entries_at_head() {
        let mut s = empty(5);
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("a".to_string())) {
            s = state;
        }
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("b".to_string())) {
            assert_eq!(state.entries[0], "b");
        }
    }

    #[test]
    fn paste_does_not_modify_state() {
        let mut s = empty(5);
        s.entries = vec!["x".to_string()];
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Paste) {
            assert_eq!(state.entries, vec!["x".to_string()]);
        }
    }

    #[test]
    fn unicode_clipboard() {
        let s = empty(5);
        if let ClipVerdict::Ok { state, .. } = apply(&s, ClipOp::Copy("café".to_string())) {
            assert_eq!(state.entries, vec!["café".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let s = empty(5);
        let a = apply(&s, ClipOp::Copy("x".to_string()));
        let b = apply(&s, ClipOp::Copy("x".to_string()));
        assert_eq!(a, b);
    }
}
