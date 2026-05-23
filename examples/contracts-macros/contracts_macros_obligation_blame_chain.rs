//! # Contracts-Macros Obligation Blame Chain
//!
//! Track who introduced each obligation across history events. Returns
//! per-obligation original-author + last-modifier maps.
//!
//! Demonstrates the **CMM.133** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: `git blame` annotation; SCM blame/annotate conventions.
//!
//! Run with: cargo run --example contracts_macros_obligation_blame_chain
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum BlameVerdict {
    Ok {
        original_author: BTreeMap<String, String>,
        last_modifier: BTreeMap<String, String>,
        modified_count: u32,
    },
    InvalidConfig,
}

pub fn build(events: &[(&str, &str, u64)]) -> BlameVerdict {
    if events.is_empty() {
        return BlameVerdict::InvalidConfig;
    }
    let mut original_author: BTreeMap<String, String> = BTreeMap::new();
    let mut last_modifier: BTreeMap<String, String> = BTreeMap::new();
    let mut last_ts: BTreeMap<String, u64> = BTreeMap::new();
    let mut modified_count = 0u32;
    for (id, author, ts) in events {
        if original_author.contains_key(*id) {
            modified_count += 1;
        } else {
            original_author.insert((*id).to_string(), (*author).to_string());
        }
        let cur_ts = last_ts.get(*id).copied().unwrap_or(0);
        if *ts >= cur_ts {
            last_modifier.insert((*id).to_string(), (*author).to_string());
            last_ts.insert((*id).to_string(), *ts);
        }
    }
    BlameVerdict::Ok {
        original_author,
        last_modifier,
        modified_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_blame_chain")?;

    let events = [
        ("o1", "alice", 100),
        ("o1", "bob", 200),
        ("o2", "carol", 150),
    ];
    println!("blame: {:?}", build(&events));
    println!("invalid: {:?}", build(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_author_is_original() {
        let events = [("o", "alice", 100)];
        let v = build(&events);
        if let BlameVerdict::Ok {
            original_author, ..
        } = v
        {
            assert_eq!(original_author.get("o"), Some(&"alice".to_string()));
        }
    }

    #[test]
    fn last_modifier_tracks_latest() {
        let events = [("o", "alice", 100), ("o", "bob", 200)];
        let v = build(&events);
        if let BlameVerdict::Ok { last_modifier, .. } = v {
            assert_eq!(last_modifier.get("o"), Some(&"bob".to_string()));
        }
    }

    #[test]
    fn modified_count_correct() {
        let events = [("o", "a", 100), ("o", "b", 200), ("o", "c", 300)];
        let v = build(&events);
        if let BlameVerdict::Ok { modified_count, .. } = v {
            assert_eq!(modified_count, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(build(&[]), BlameVerdict::InvalidConfig);
    }

    #[test]
    fn original_unchanged_by_modifications() {
        let events = [("o", "alice", 100), ("o", "bob", 200)];
        let v = build(&events);
        if let BlameVerdict::Ok {
            original_author, ..
        } = v
        {
            assert_eq!(original_author.get("o"), Some(&"alice".to_string()));
        }
    }

    #[test]
    fn deterministic() {
        let events = [("o", "a", 100)];
        let r1 = build(&events);
        let r2 = build(&events);
        assert_eq!(r1, r2);
    }

    #[test]
    fn distinct_obligations_independent() {
        let events = [("a", "alice", 100), ("b", "bob", 200)];
        let v = build(&events);
        if let BlameVerdict::Ok {
            original_author, ..
        } = v
        {
            assert_eq!(original_author.get("a"), Some(&"alice".to_string()));
            assert_eq!(original_author.get("b"), Some(&"bob".to_string()));
        }
    }

    #[test]
    fn out_of_order_timestamps_handled() {
        let events = [("o", "a", 200), ("o", "b", 100)];
        let v = build(&events);
        if let BlameVerdict::Ok { last_modifier, .. } = v {
            // Latest timestamp wins regardless of event order.
            assert_eq!(last_modifier.get("o"), Some(&"a".to_string()));
        }
    }

    #[test]
    fn many_events_handled() {
        let events: Vec<(&str, &str, u64)> = (0..30).map(|i| ("o", "alice", i)).collect();
        let v = build(&events);
        if let BlameVerdict::Ok { modified_count, .. } = v {
            assert_eq!(modified_count, 29);
        }
    }

    #[test]
    fn no_modifications_zero_count() {
        let events = [("o", "alice", 100)];
        let v = build(&events);
        if let BlameVerdict::Ok { modified_count, .. } = v {
            assert_eq!(modified_count, 0);
        }
    }

    #[test]
    fn unicode_author_supported() {
        let events = [("o", "café", 100)];
        let v = build(&events);
        if let BlameVerdict::Ok {
            original_author, ..
        } = v
        {
            assert_eq!(original_author.get("o"), Some(&"café".to_string()));
        }
    }
}
