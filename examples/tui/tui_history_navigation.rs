//! # TUI History Navigation
//!
//! Maintain a back/forward history stack like a web browser:
//! `Visit(url)` truncates forward; `Back` and `Forward` navigate.
//! Returns the new state and current location.
//!
//! Demonstrates the **TUI.39** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 History API behavior.
//!
//! Run with: cargo run --example tui_history_navigation
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryState {
    pub entries: Vec<String>,
    pub current: usize,
    pub max_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryOp {
    Visit(String),
    Back,
    Forward,
}

#[derive(Debug, PartialEq)]
pub enum HistoryVerdict {
    Ok {
        state: HistoryState,
        current: Option<String>,
    },
    NoBack,
    NoForward,
    InvalidConfig,
}

pub fn apply(state: &HistoryState, op: HistoryOp) -> HistoryVerdict {
    if state.max_size == 0 {
        return HistoryVerdict::InvalidConfig;
    }
    match op {
        HistoryOp::Visit(url) => {
            let mut new_entries = state.entries.clone();
            // Truncate forward on visit.
            if !new_entries.is_empty() && state.current + 1 < new_entries.len() {
                new_entries.truncate(state.current + 1);
            }
            new_entries.push(url);
            // Cap at max_size by dropping oldest.
            while new_entries.len() > state.max_size {
                new_entries.remove(0);
            }
            let current = new_entries.len() - 1;
            HistoryVerdict::Ok {
                state: HistoryState {
                    current,
                    entries: new_entries.clone(),
                    max_size: state.max_size,
                },
                current: new_entries.last().cloned(),
            }
        }
        HistoryOp::Back => {
            if state.current == 0 || state.entries.is_empty() {
                return HistoryVerdict::NoBack;
            }
            let new_current = state.current - 1;
            HistoryVerdict::Ok {
                current: state.entries.get(new_current).cloned(),
                state: HistoryState {
                    current: new_current,
                    ..state.clone()
                },
            }
        }
        HistoryOp::Forward => {
            if state.current + 1 >= state.entries.len() {
                return HistoryVerdict::NoForward;
            }
            let new_current = state.current + 1;
            HistoryVerdict::Ok {
                current: state.entries.get(new_current).cloned(),
                state: HistoryState {
                    current: new_current,
                    ..state.clone()
                },
            }
        }
    }
}

fn empty(max_size: usize) -> HistoryState {
    HistoryState {
        entries: Vec::new(),
        current: 0,
        max_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_history_navigation")?;

    let s0 = empty(10);
    let s1 = if let HistoryVerdict::Ok { state, .. } = apply(&s0, HistoryOp::Visit("a".to_string()))
    {
        state
    } else {
        s0.clone()
    };
    let s2 = if let HistoryVerdict::Ok { state, .. } = apply(&s1, HistoryOp::Visit("b".to_string()))
    {
        state
    } else {
        s1.clone()
    };
    println!("after visits: {s2:?}");
    let s3 = if let HistoryVerdict::Ok { state, current } = apply(&s2, HistoryOp::Back) {
        println!("back: {current:?}");
        state
    } else {
        s2.clone()
    };
    let _s4 = if let HistoryVerdict::Ok { state, current } = apply(&s3, HistoryOp::Forward) {
        println!("forward: {current:?}");
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
    fn navigator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn visit_appends() {
        let s = empty(10);
        let v = apply(&s, HistoryOp::Visit("a".to_string()));
        if let HistoryVerdict::Ok { state, .. } = v {
            assert_eq!(state.entries, vec!["a".to_string()]);
        }
    }

    #[test]
    fn back_at_start_no_back() {
        let s = empty(10);
        assert_eq!(apply(&s, HistoryOp::Back), HistoryVerdict::NoBack);
    }

    #[test]
    fn forward_at_end_no_forward() {
        let mut s = empty(10);
        s.entries = vec!["a".to_string()];
        s.current = 0;
        assert_eq!(apply(&s, HistoryOp::Forward), HistoryVerdict::NoForward);
    }

    #[test]
    fn back_decrements_current() {
        let mut s = empty(10);
        s.entries = vec!["a".to_string(), "b".to_string()];
        s.current = 1;
        let v = apply(&s, HistoryOp::Back);
        if let HistoryVerdict::Ok { state, .. } = v {
            assert_eq!(state.current, 0);
        }
    }

    #[test]
    fn visit_truncates_forward() {
        let mut s = empty(10);
        s.entries = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        s.current = 0;
        let v = apply(&s, HistoryOp::Visit("d".to_string()));
        if let HistoryVerdict::Ok { state, .. } = v {
            // After visit, forward (b, c) is truncated.
            assert_eq!(state.entries, vec!["a".to_string(), "d".to_string()]);
        }
    }

    #[test]
    fn invalid_zero_max_size() {
        let s = empty(0);
        assert_eq!(
            apply(&s, HistoryOp::Visit("x".to_string())),
            HistoryVerdict::InvalidConfig
        );
    }

    #[test]
    fn capacity_drops_oldest() {
        let mut s = empty(2);
        if let HistoryVerdict::Ok { state, .. } = apply(&s, HistoryOp::Visit("a".to_string())) {
            s = state;
        }
        if let HistoryVerdict::Ok { state, .. } = apply(&s, HistoryOp::Visit("b".to_string())) {
            s = state;
        }
        if let HistoryVerdict::Ok { state, .. } = apply(&s, HistoryOp::Visit("c".to_string())) {
            assert_eq!(state.entries, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn forward_after_back_works() {
        let mut s = empty(10);
        s.entries = vec!["a".to_string(), "b".to_string()];
        s.current = 0;
        let v = apply(&s, HistoryOp::Forward);
        if let HistoryVerdict::Ok { state, .. } = v {
            assert_eq!(state.current, 1);
        }
    }

    #[test]
    fn current_returned() {
        let s = empty(10);
        let v = apply(&s, HistoryOp::Visit("a".to_string()));
        if let HistoryVerdict::Ok { current, .. } = v {
            assert_eq!(current, Some("a".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let s = empty(10);
        let a = apply(&s, HistoryOp::Visit("x".to_string()));
        let b = apply(&s, HistoryOp::Visit("x".to_string()));
        assert_eq!(a, b);
    }
}
