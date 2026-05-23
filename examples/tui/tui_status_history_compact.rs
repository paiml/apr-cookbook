//! # TUI Status History Compact
//!
//! Show the N most recent status messages, dropping older ones.
//! Returns visible list (newest first) and dropped count.
//!
//! Demonstrates the **TUI.107** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: tail -n streaming convention; ELK Kibana log viewer
//!  N-most-recent display.
//!
//! Run with: cargo run --example tui_status_history_compact
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompactVerdict {
    Ok {
        visible: Vec<String>,
        dropped_count: u32,
    },
    InvalidConfig,
}

pub fn compact(messages: &[&str], visible_n: u32) -> CompactVerdict {
    if visible_n == 0 {
        return CompactVerdict::InvalidConfig;
    }
    let total = messages.len();
    if total <= visible_n as usize {
        let mut visible: Vec<String> = messages.iter().map(|s| (*s).to_string()).collect();
        visible.reverse();
        return CompactVerdict::Ok {
            visible,
            dropped_count: 0,
        };
    }
    let start = total - visible_n as usize;
    let mut visible: Vec<String> = messages[start..].iter().map(|s| (*s).to_string()).collect();
    visible.reverse();
    let dropped_count = (total - visible_n as usize) as u32;
    CompactVerdict::Ok {
        visible,
        dropped_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_status_history_compact")?;

    let msgs = ["Started", "Loaded", "Connected", "Synced", "Done"];
    println!("show 3: {:?}", compact(&msgs, 3));
    println!("show 10: {:?}", compact(&msgs, 10));
    println!("invalid: {:?}", compact(&msgs, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compactor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_limit_no_drops() {
        let msgs = ["a", "b"];
        let v = compact(&msgs, 5);
        if let CompactVerdict::Ok {
            visible,
            dropped_count,
        } = v
        {
            assert_eq!(visible, vec!["b".to_string(), "a".to_string()]);
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn over_limit_drops_oldest() {
        let msgs = ["a", "b", "c", "d"];
        let v = compact(&msgs, 2);
        if let CompactVerdict::Ok {
            visible,
            dropped_count,
        } = v
        {
            assert_eq!(visible, vec!["d".to_string(), "c".to_string()]);
            assert_eq!(dropped_count, 2);
        }
    }

    #[test]
    fn empty_messages_no_drops() {
        let v = compact(&[], 5);
        if let CompactVerdict::Ok {
            visible,
            dropped_count,
        } = v
        {
            assert!(visible.is_empty());
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn zero_visible_rejected() {
        assert_eq!(compact(&["a"], 0), CompactVerdict::InvalidConfig);
    }

    #[test]
    fn newest_first_order() {
        let msgs = ["a", "b", "c"];
        let v = compact(&msgs, 5);
        if let CompactVerdict::Ok { visible, .. } = v {
            assert_eq!(visible[0], "c");
            assert_eq!(visible.last(), Some(&"a".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let msgs = ["a", "b"];
        let r1 = compact(&msgs, 5);
        let r2 = compact(&msgs, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn visible_count_le_n() {
        let msgs = ["a", "b", "c"];
        let v = compact(&msgs, 2);
        if let CompactVerdict::Ok { visible, .. } = v {
            assert_eq!(visible.len(), 2);
        }
    }

    #[test]
    fn n_one_only_newest() {
        let msgs = ["a", "b", "c"];
        let v = compact(&msgs, 1);
        if let CompactVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["c".to_string()]);
        }
    }

    #[test]
    fn visible_plus_dropped_equals_total() {
        let msgs = ["a", "b", "c", "d", "e"];
        let v = compact(&msgs, 3);
        if let CompactVerdict::Ok {
            visible,
            dropped_count,
        } = v
        {
            assert_eq!(visible.len() + dropped_count as usize, 5);
        }
    }

    #[test]
    fn unicode_messages_supported() {
        let msgs = ["café", "résumé"];
        let v = compact(&msgs, 5);
        if let CompactVerdict::Ok { visible, .. } = v {
            assert_eq!(visible[0], "résumé");
        }
    }

    #[test]
    fn many_messages_handled() {
        let msgs: Vec<&str> = vec!["m"; 100];
        let v = compact(&msgs, 10);
        if let CompactVerdict::Ok {
            visible,
            dropped_count,
        } = v
        {
            assert_eq!(visible.len(), 10);
            assert_eq!(dropped_count, 90);
        }
    }
}
