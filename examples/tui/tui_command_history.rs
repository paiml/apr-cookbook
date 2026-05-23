//! # TUI Command History
//!
//! Readline-style command history: append, navigate up/down, and
//! retrieve the entry at the current cursor. Returns the new state.
//!
//! Demonstrates the **TUI.45** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU readline history (HISTFILE behavior).
//!
//! Run with: cargo run --example tui_command_history
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct History {
    pub entries: Vec<String>,
    pub cursor: Option<usize>,
    pub max_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistOp {
    Append(String),
    Up,
    Down,
}

#[derive(Debug, PartialEq)]
pub enum HistVerdict {
    Ok {
        state: History,
        current: Option<String>,
    },
    InvalidConfig,
}

pub fn apply(state: &History, op: HistOp) -> HistVerdict {
    if state.max_size == 0 {
        return HistVerdict::InvalidConfig;
    }
    match op {
        HistOp::Append(line) => {
            let mut new_entries = state.entries.clone();
            // Skip exact duplicates of last entry (readline default).
            if new_entries.last() != Some(&line) {
                new_entries.push(line);
            }
            while new_entries.len() > state.max_size {
                new_entries.remove(0);
            }
            HistVerdict::Ok {
                state: History {
                    entries: new_entries,
                    cursor: None,
                    max_size: state.max_size,
                },
                current: None,
            }
        }
        HistOp::Up => {
            if state.entries.is_empty() {
                return HistVerdict::Ok {
                    state: state.clone(),
                    current: None,
                };
            }
            let new_cursor = match state.cursor {
                None => state.entries.len() - 1,
                Some(0) => 0,
                Some(c) => c - 1,
            };
            HistVerdict::Ok {
                current: state.entries.get(new_cursor).cloned(),
                state: History {
                    cursor: Some(new_cursor),
                    ..state.clone()
                },
            }
        }
        HistOp::Down => {
            let Some(c) = state.cursor else {
                return HistVerdict::Ok {
                    state: state.clone(),
                    current: None,
                };
            };
            let last = state.entries.len().saturating_sub(1);
            if c >= last {
                return HistVerdict::Ok {
                    state: History {
                        cursor: None,
                        ..state.clone()
                    },
                    current: None,
                };
            }
            let new_cursor = c + 1;
            HistVerdict::Ok {
                current: state.entries.get(new_cursor).cloned(),
                state: History {
                    cursor: Some(new_cursor),
                    ..state.clone()
                },
            }
        }
    }
}

fn empty(max_size: usize) -> History {
    History {
        entries: Vec::new(),
        cursor: None,
        max_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_command_history")?;

    let s0 = empty(10);
    let s1 = if let HistVerdict::Ok { state, .. } = apply(&s0, HistOp::Append("ls".to_string())) {
        state
    } else {
        s0.clone()
    };
    let s2 = if let HistVerdict::Ok { state, .. } = apply(&s1, HistOp::Append("cd".to_string())) {
        state
    } else {
        s1.clone()
    };
    let s3 = if let HistVerdict::Ok { state, current } = apply(&s2, HistOp::Up) {
        println!("up: {current:?}");
        state
    } else {
        s2.clone()
    };
    let _s4 = if let HistVerdict::Ok { state, current } = apply(&s3, HistOp::Up) {
        println!("up again: {current:?}");
        state
    } else {
        s3.clone()
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn append_grows_entries() {
        let s = empty(5);
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("a".to_string())) {
            assert_eq!(state.entries, vec!["a".to_string()]);
        }
    }

    #[test]
    fn up_starts_at_last() {
        let mut s = empty(5);
        s.entries = vec!["a".to_string(), "b".to_string()];
        if let HistVerdict::Ok { current, .. } = apply(&s, HistOp::Up) {
            assert_eq!(current, Some("b".to_string()));
        }
    }

    #[test]
    fn up_at_first_clamps() {
        let mut s = empty(5);
        s.entries = vec!["a".to_string()];
        s.cursor = Some(0);
        if let HistVerdict::Ok { current, .. } = apply(&s, HistOp::Up) {
            assert_eq!(current, Some("a".to_string()));
        }
    }

    #[test]
    fn down_past_last_clears_cursor() {
        let mut s = empty(5);
        s.entries = vec!["a".to_string()];
        s.cursor = Some(0);
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Down) {
            assert_eq!(state.cursor, None);
        }
    }

    #[test]
    fn duplicate_append_skipped() {
        let mut s = empty(5);
        s.entries = vec!["a".to_string()];
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("a".to_string())) {
            assert_eq!(state.entries.len(), 1);
        }
    }

    #[test]
    fn capacity_drops_oldest() {
        let mut s = empty(2);
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("a".to_string())) {
            s = state;
        }
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("b".to_string())) {
            s = state;
        }
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("c".to_string())) {
            assert_eq!(state.entries, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn invalid_zero_max() {
        let s = empty(0);
        assert_eq!(
            apply(&s, HistOp::Append("x".to_string())),
            HistVerdict::InvalidConfig
        );
    }

    #[test]
    fn append_resets_cursor() {
        let mut s = empty(5);
        s.entries = vec!["a".to_string()];
        s.cursor = Some(0);
        if let HistVerdict::Ok { state, .. } = apply(&s, HistOp::Append("b".to_string())) {
            assert_eq!(state.cursor, None);
        }
    }

    #[test]
    fn up_on_empty_no_cursor() {
        let s = empty(5);
        if let HistVerdict::Ok { current, .. } = apply(&s, HistOp::Up) {
            assert_eq!(current, None);
        }
    }

    #[test]
    fn deterministic() {
        let s = empty(5);
        let a = apply(&s, HistOp::Append("x".to_string()));
        let b = apply(&s, HistOp::Append("x".to_string()));
        assert_eq!(a, b);
    }
}
