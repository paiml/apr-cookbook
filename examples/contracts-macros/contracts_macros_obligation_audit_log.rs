//! # Contracts-Macros Obligation Audit Log
//!
//! Generate an append-only audit log entry per obligation status
//! change. Entries are formatted as `<timestamp>|<actor>|<from>|<to>`.
//!
//! Demonstrates the **CMM.115** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SOX audit-trail conventions; AWS CloudTrail event format.
//!
//! Run with: cargo run --example contracts_macros_obligation_audit_log
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AuditLogVerdict {
    Ok {
        entries: Vec<String>,
        chronological: bool,
    },
    InvalidConfig,
}

pub fn build(events: &[(u64, &str, &str, &str)]) -> AuditLogVerdict {
    if events.is_empty() {
        return AuditLogVerdict::InvalidConfig;
    }
    let mut entries: Vec<String> = Vec::with_capacity(events.len());
    let mut chronological = true;
    let mut prev_ts: u64 = 0;
    for (ts, actor, from, to) in events {
        if *ts < prev_ts {
            chronological = false;
        }
        prev_ts = *ts;
        entries.push(format!("{ts}|{actor}|{from}|{to}"));
    }
    AuditLogVerdict::Ok {
        entries,
        chronological,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_audit_log")?;

    let events = [
        (1u64, "alice", "draft", "open"),
        (2, "bob", "open", "implemented"),
        (3, "carol", "implemented", "closed"),
    ];
    println!("ordered: {:?}", build(&events));
    let bad = [(2u64, "alice", "x", "y"), (1, "bob", "y", "z")];
    println!("out of order: {:?}", build(&bad));
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
    fn entries_match_event_count() {
        let events = [(1u64, "a", "x", "y")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert_eq!(entries.len(), 1);
        }
    }

    #[test]
    fn chronological_ordering_detected() {
        let events = [(1u64, "a", "x", "y"), (2, "b", "y", "z")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { chronological, .. } = v {
            assert!(chronological);
        }
    }

    #[test]
    fn out_of_order_detected() {
        let events = [(2u64, "a", "x", "y"), (1, "b", "y", "z")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { chronological, .. } = v {
            assert!(!chronological);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(build(&[]), AuditLogVerdict::InvalidConfig);
    }

    #[test]
    fn entry_format_pipe_separated() {
        let events = [(1u64, "a", "x", "y")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert_eq!(entries[0].matches('|').count(), 3);
        }
    }

    #[test]
    fn deterministic() {
        let events = [(1u64, "a", "x", "y")];
        let r1 = build(&events);
        let r2 = build(&events);
        assert_eq!(r1, r2);
    }

    #[test]
    fn equal_timestamps_chronological() {
        let events = [(1u64, "a", "x", "y"), (1, "b", "y", "z")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { chronological, .. } = v {
            assert!(chronological);
        }
    }

    #[test]
    fn entry_includes_all_fields() {
        let events = [(42u64, "alice", "draft", "open")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert!(entries[0].contains("42"));
            assert!(entries[0].contains("alice"));
            assert!(entries[0].contains("draft"));
            assert!(entries[0].contains("open"));
        }
    }

    #[test]
    fn many_events_handled() {
        let events: Vec<(u64, &str, &str, &str)> = (0..100).map(|i| (i, "a", "x", "y")).collect();
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert_eq!(entries.len(), 100);
        }
    }

    #[test]
    fn unicode_actor_supported() {
        let events = [(1u64, "café", "x", "y")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert!(entries[0].contains("café"));
        }
    }

    #[test]
    fn order_preserved_in_entries() {
        let events = [(1u64, "first", "x", "y"), (2, "second", "y", "z")];
        let v = build(&events);
        if let AuditLogVerdict::Ok { entries, .. } = v {
            assert!(entries[0].contains("first"));
            assert!(entries[1].contains("second"));
        }
    }
}
