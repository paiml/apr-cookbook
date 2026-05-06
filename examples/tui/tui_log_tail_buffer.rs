//! # TUI Log Tail Buffer
//!
//! Maintain a circular buffer of the last N log lines. Returns the
//! current visible window (most recent first or insertion order).
//! Pure functional: returns new buffer state.
//!
//! Demonstrates the **TUI.27** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ring-buffer tail logging (journalctl, tmux scrollback).
//!
//! Run with: cargo run --example tui_log_tail_buffer
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BufferVerdict {
    Ok { lines: Vec<String>, dropped: u32 },
    InvalidCapacity,
}

pub fn tail(input_lines: &[&str], capacity: usize) -> BufferVerdict {
    if capacity == 0 {
        return BufferVerdict::InvalidCapacity;
    }
    let n = input_lines.len();
    let dropped = if n > capacity {
        (n - capacity) as u32
    } else {
        0
    };
    let start = n.saturating_sub(capacity);
    let lines: Vec<String> = input_lines[start..]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    BufferVerdict::Ok { lines, dropped }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_log_tail_buffer")?;

    let logs = ["line 1", "line 2", "line 3", "line 4", "line 5"];
    println!("capacity 3: {:?}", tail(&logs, 3));
    println!("capacity 10: {:?}", tail(&logs, 10));
    println!("empty: {:?}", tail(&[], 5));
    println!("invalid: {:?}", tail(&logs, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tailer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn keeps_only_last_n() {
        let logs = ["a", "b", "c", "d", "e"];
        let v = tail(&logs, 3);
        if let BufferVerdict::Ok { lines, dropped } = v {
            assert_eq!(
                lines,
                vec!["c".to_string(), "d".to_string(), "e".to_string()]
            );
            assert_eq!(dropped, 2);
        }
    }

    #[test]
    fn capacity_larger_than_input() {
        let logs = ["a", "b"];
        let v = tail(&logs, 10);
        if let BufferVerdict::Ok { lines, dropped } = v {
            assert_eq!(lines.len(), 2);
            assert_eq!(dropped, 0);
        }
    }

    #[test]
    fn empty_input_works() {
        let v = tail(&[], 5);
        if let BufferVerdict::Ok { lines, dropped } = v {
            assert!(lines.is_empty());
            assert_eq!(dropped, 0);
        }
    }

    #[test]
    fn zero_capacity_invalid() {
        let logs = ["a"];
        assert_eq!(tail(&logs, 0), BufferVerdict::InvalidCapacity);
    }

    #[test]
    fn capacity_one_keeps_last() {
        let logs = ["a", "b", "c"];
        let v = tail(&logs, 1);
        if let BufferVerdict::Ok { lines, dropped } = v {
            assert_eq!(lines, vec!["c".to_string()]);
            assert_eq!(dropped, 2);
        }
    }

    #[test]
    fn order_preserved() {
        let logs = ["a", "b", "c"];
        let v = tail(&logs, 5);
        if let BufferVerdict::Ok { lines, .. } = v {
            assert_eq!(
                lines,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
    }

    #[test]
    fn unicode_lines() {
        let logs = ["héllo", "wörld"];
        let v = tail(&logs, 5);
        if let BufferVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 2);
        }
    }

    #[test]
    fn many_logs() {
        let logs: Vec<&str> = (0..1000).map(|_| "x").collect();
        let v = tail(&logs, 100);
        if let BufferVerdict::Ok { lines, dropped } = v {
            assert_eq!(lines.len(), 100);
            assert_eq!(dropped, 900);
        }
    }

    #[test]
    fn capacity_equal_input_no_drop() {
        let logs = ["a", "b", "c"];
        let v = tail(&logs, 3);
        if let BufferVerdict::Ok { dropped, .. } = v {
            assert_eq!(dropped, 0);
        }
    }

    #[test]
    fn deterministic() {
        let logs = ["a", "b", "c"];
        let a = tail(&logs, 2);
        let b = tail(&logs, 2);
        assert_eq!(a, b);
    }
}
