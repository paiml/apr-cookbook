//! # TUI Clipboard History Ring Buffer
//!
//! Maintain a circular history of the last `capacity` clipboard
//! entries. Pushing past capacity evicts oldest. Returns history
//! list (newest first) and evicted count.
//!
//! Demonstrates the **TUI.91** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ring-buffer abstract data type (Knuth, TAOCP §2.2.5);
//!  macOS Mission Control clipboard history.
//!
//! Run with: cargo run --example tui_clipboard_history_circular
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HistoryVerdict {
    Ok {
        history: Vec<String>,
        evicted_count: u32,
    },
    InvalidConfig,
}

pub fn ingest(entries: &[&str], capacity: u32) -> HistoryVerdict {
    if capacity == 0 {
        return HistoryVerdict::InvalidConfig;
    }
    let cap = capacity as usize;
    let mut buffer: Vec<String> = Vec::with_capacity(cap);
    let mut evicted: u32 = 0;
    for entry in entries {
        if buffer.len() >= cap {
            buffer.remove(0);
            evicted += 1;
        }
        buffer.push((*entry).to_string());
    }
    buffer.reverse();
    HistoryVerdict::Ok {
        history: buffer,
        evicted_count: evicted,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_clipboard_history_circular")?;

    let entries = ["a", "b", "c", "d", "e"];
    println!("cap=3: {:?}", ingest(&entries, 3));
    println!("cap=10: {:?}", ingest(&entries, 10));
    println!("invalid: {:?}", ingest(&[], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_capacity_no_eviction() {
        let entries = ["a", "b"];
        let v = ingest(&entries, 5);
        if let HistoryVerdict::Ok {
            history,
            evicted_count,
        } = v
        {
            assert_eq!(history, vec!["b".to_string(), "a".to_string()]);
            assert_eq!(evicted_count, 0);
        }
    }

    #[test]
    fn over_capacity_evicts_oldest() {
        let entries = ["a", "b", "c", "d"];
        let v = ingest(&entries, 2);
        if let HistoryVerdict::Ok {
            history,
            evicted_count,
        } = v
        {
            assert_eq!(history, vec!["d".to_string(), "c".to_string()]);
            assert_eq!(evicted_count, 2);
        }
    }

    #[test]
    fn empty_entries_empty_history() {
        let v = ingest(&[], 5);
        if let HistoryVerdict::Ok {
            history,
            evicted_count,
        } = v
        {
            assert!(history.is_empty());
            assert_eq!(evicted_count, 0);
        }
    }

    #[test]
    fn zero_capacity_rejected() {
        assert_eq!(ingest(&["a"], 0), HistoryVerdict::InvalidConfig);
    }

    #[test]
    fn newest_first_order() {
        let entries = ["a", "b", "c"];
        let v = ingest(&entries, 5);
        if let HistoryVerdict::Ok { history, .. } = v {
            assert_eq!(history[0], "c");
            assert_eq!(history.last(), Some(&"a".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let entries = ["a", "b"];
        let r1 = ingest(&entries, 5);
        let r2 = ingest(&entries, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn cap_1_only_keeps_latest() {
        let entries = ["a", "b", "c"];
        let v = ingest(&entries, 1);
        if let HistoryVerdict::Ok {
            history,
            evicted_count,
        } = v
        {
            assert_eq!(history, vec!["c".to_string()]);
            assert_eq!(evicted_count, 2);
        }
    }

    #[test]
    fn duplicate_entries_kept() {
        let entries = ["a", "a", "a"];
        let v = ingest(&entries, 5);
        if let HistoryVerdict::Ok { history, .. } = v {
            assert_eq!(history.len(), 3);
        }
    }

    #[test]
    fn history_size_le_capacity() {
        let entries = ["a", "b", "c", "d", "e"];
        let v = ingest(&entries, 3);
        if let HistoryVerdict::Ok { history, .. } = v {
            assert!(history.len() <= 3);
        }
    }

    #[test]
    fn unicode_entries_supported() {
        let entries = ["café"];
        let v = ingest(&entries, 5);
        if let HistoryVerdict::Ok { history, .. } = v {
            assert_eq!(history, vec!["café".to_string()]);
        }
    }

    #[test]
    fn many_entries_handled() {
        let entries: Vec<&str> = vec!["x"; 100];
        let v = ingest(&entries, 10);
        if let HistoryVerdict::Ok {
            history,
            evicted_count,
        } = v
        {
            assert_eq!(history.len(), 10);
            assert_eq!(evicted_count, 90);
        }
    }
}
