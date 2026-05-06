//! # TUI Input Buffer State
//!
//! Maintain an editable input buffer: handles insert at cursor,
//! backspace, and cursor moves. Returns the new state.
//!
//! Demonstrates the **TUI.09** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU readline editing primitives.
//!
//! Run with: cargo run --example tui_input_buffer
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferState {
    pub text: String,
    pub cursor: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferOp {
    InsertChar(char),
    Backspace,
    Delete,
    MoveLeft,
    MoveRight,
    Home,
    End,
}

pub fn apply(state: &BufferState, op: BufferOp) -> BufferState {
    let mut chars: Vec<char> = state.text.chars().collect();
    let mut cursor = state.cursor.min(chars.len());
    match op {
        BufferOp::InsertChar(c) => {
            chars.insert(cursor, c);
            cursor += 1;
        }
        BufferOp::Backspace => {
            if cursor > 0 {
                chars.remove(cursor - 1);
                cursor -= 1;
            }
        }
        BufferOp::Delete => {
            if cursor < chars.len() {
                chars.remove(cursor);
            }
        }
        BufferOp::MoveLeft => cursor = cursor.saturating_sub(1),
        BufferOp::MoveRight => cursor = (cursor + 1).min(chars.len()),
        BufferOp::Home => cursor = 0,
        BufferOp::End => cursor = chars.len(),
    }
    BufferState {
        text: chars.iter().collect(),
        cursor,
    }
}

fn empty() -> BufferState {
    BufferState {
        text: String::new(),
        cursor: 0,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_input_buffer")?;

    let s0 = empty();
    let s1 = apply(&s0, BufferOp::InsertChar('h'));
    let s2 = apply(&s1, BufferOp::InsertChar('i'));
    let s3 = apply(&s2, BufferOp::Home);
    let s4 = apply(&s3, BufferOp::InsertChar('H'));
    println!("h: {s1:?}");
    println!("hi: {s2:?}");
    println!("home: {s3:?}");
    println!("Hhi: {s4:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(text: &str, cursor: usize) -> BufferState {
        BufferState {
            text: text.to_string(),
            cursor,
        }
    }

    #[test]
    fn buffer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn insert_at_cursor() {
        let s = apply(&buf("h", 1), BufferOp::InsertChar('i'));
        assert_eq!(s, buf("hi", 2));
    }

    #[test]
    fn insert_at_start() {
        let s = apply(&buf("ello", 0), BufferOp::InsertChar('h'));
        assert_eq!(s, buf("hello", 1));
    }

    #[test]
    fn backspace_removes_left_char() {
        let s = apply(&buf("hi", 2), BufferOp::Backspace);
        assert_eq!(s, buf("h", 1));
    }

    #[test]
    fn backspace_at_start_no_change() {
        let s = apply(&buf("hi", 0), BufferOp::Backspace);
        assert_eq!(s, buf("hi", 0));
    }

    #[test]
    fn delete_removes_right_char() {
        let s = apply(&buf("hi", 0), BufferOp::Delete);
        assert_eq!(s, buf("i", 0));
    }

    #[test]
    fn delete_at_end_no_change() {
        let s = apply(&buf("hi", 2), BufferOp::Delete);
        assert_eq!(s, buf("hi", 2));
    }

    #[test]
    fn move_left() {
        let s = apply(&buf("hi", 2), BufferOp::MoveLeft);
        assert_eq!(s, buf("hi", 1));
    }

    #[test]
    fn move_right_clamps() {
        let s = apply(&buf("hi", 2), BufferOp::MoveRight);
        assert_eq!(s, buf("hi", 2));
    }

    #[test]
    fn home_to_zero() {
        let s = apply(&buf("hello", 3), BufferOp::Home);
        assert_eq!(s, buf("hello", 0));
    }

    #[test]
    fn end_to_len() {
        let s = apply(&buf("hello", 0), BufferOp::End);
        assert_eq!(s, buf("hello", 5));
    }

    #[test]
    fn unicode_insert() {
        let s = apply(&buf("h", 1), BufferOp::InsertChar('é'));
        assert_eq!(s, buf("hé", 2));
    }

    #[test]
    fn cursor_clamps_in_apply() {
        // Cursor past end is clamped before op.
        let s = apply(&buf("hi", 100), BufferOp::Backspace);
        assert_eq!(s, buf("h", 1));
    }

    #[test]
    fn deterministic() {
        let s0 = buf("hi", 1);
        let a = apply(&s0, BufferOp::InsertChar('x'));
        let b = apply(&s0, BufferOp::InsertChar('x'));
        assert_eq!(a, b);
    }
}
