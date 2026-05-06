//! # Distributed Split-Brain Detector
//!
//! Multiple nodes claim leader → split-brain. Detection rules:
//!   exactly one leader heartbeat in window → Single
//!   multiple distinct leader IDs → SplitBrain
//!   no leader heartbeats → NoLeader (election needed)
//!
//! Plus stale-leader: leader heartbeat older than max_silence → also
//! NoLeader.
//!
//! Demonstrates the **DIST.14** recipe for PMAT-145 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Raft paper § Safety (no two leaders in same term).
//!
//! Run with: cargo run --example distributed_split_brain_detector
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum LeaderVerdict {
    SingleLeader { id: String, term: u64 },
    SplitBrain { leaders: Vec<String> },
    NoLeader,
    InvalidWindow,
}

#[derive(Debug, Clone)]
pub struct LeaderHeartbeat {
    pub leader_id: String,
    pub term: u64,
    pub received_at_secs: u64,
}

pub fn check(
    heartbeats: &[LeaderHeartbeat],
    now_secs: u64,
    max_silence_secs: u64,
) -> LeaderVerdict {
    if max_silence_secs == 0 {
        return LeaderVerdict::InvalidWindow;
    }
    let fresh: Vec<&LeaderHeartbeat> = heartbeats
        .iter()
        .filter(|hb| now_secs.saturating_sub(hb.received_at_secs) <= max_silence_secs)
        .collect();
    if fresh.is_empty() {
        return LeaderVerdict::NoLeader;
    }
    let unique_ids: BTreeSet<&str> = fresh.iter().map(|hb| hb.leader_id.as_str()).collect();
    if unique_ids.len() == 1 {
        let highest = fresh.iter().max_by_key(|hb| hb.term).unwrap();
        return LeaderVerdict::SingleLeader {
            id: highest.leader_id.clone(),
            term: highest.term,
        };
    }
    LeaderVerdict::SplitBrain {
        leaders: unique_ids.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_split_brain_detector")?;

    let single = vec![LeaderHeartbeat {
        leader_id: "node-a".to_string(),
        term: 5,
        received_at_secs: 100,
    }];
    println!("single leader: {:?}", check(&single, 105, 30));

    let split = vec![
        LeaderHeartbeat {
            leader_id: "node-a".to_string(),
            term: 5,
            received_at_secs: 100,
        },
        LeaderHeartbeat {
            leader_id: "node-b".to_string(),
            term: 5,
            received_at_secs: 102,
        },
    ];
    println!("split brain: {:?}", check(&split, 105, 30));

    let stale = vec![LeaderHeartbeat {
        leader_id: "node-a".to_string(),
        term: 5,
        received_at_secs: 100,
    }];
    println!("stale: {:?}", check(&stale, 200, 30));

    println!("invalid: {:?}", check(&[], 100, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(id: &str, term: u64, at: u64) -> LeaderHeartbeat {
        LeaderHeartbeat {
            leader_id: id.to_string(),
            term,
            received_at_secs: at,
        }
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_leader_detected() {
        let hbs = vec![hb("a", 5, 100)];
        let v = check(&hbs, 105, 30);
        assert!(matches!(v, LeaderVerdict::SingleLeader { .. }));
    }

    #[test]
    fn split_brain_detected() {
        let hbs = vec![hb("a", 5, 100), hb("b", 5, 102)];
        let v = check(&hbs, 105, 30);
        assert!(matches!(v, LeaderVerdict::SplitBrain { .. }));
    }

    #[test]
    fn no_leader_when_empty() {
        assert_eq!(check(&[], 100, 30), LeaderVerdict::NoLeader);
    }

    #[test]
    fn stale_treated_as_no_leader() {
        let hbs = vec![hb("a", 5, 50)];
        let v = check(&hbs, 200, 30);
        assert_eq!(v, LeaderVerdict::NoLeader);
    }

    #[test]
    fn fresh_only_filters_stale() {
        // Stale heartbeat from "b" should be ignored.
        let hbs = vec![hb("a", 5, 100), hb("b", 5, 50)];
        let v = check(&hbs, 105, 30);
        assert!(matches!(v, LeaderVerdict::SingleLeader { .. }));
    }

    #[test]
    fn invalid_window_zero_rejected() {
        let hbs = vec![hb("a", 5, 100)];
        assert_eq!(check(&hbs, 105, 0), LeaderVerdict::InvalidWindow);
    }

    #[test]
    fn split_brain_lists_all_leaders() {
        let hbs = vec![hb("a", 5, 100), hb("b", 5, 100), hb("c", 5, 100)];
        if let LeaderVerdict::SplitBrain { leaders } = check(&hbs, 105, 30) {
            assert_eq!(leaders.len(), 3);
        }
    }

    #[test]
    fn highest_term_wins_for_single_id() {
        let hbs = vec![hb("a", 4, 100), hb("a", 5, 102)];
        if let LeaderVerdict::SingleLeader { term, .. } = check(&hbs, 105, 30) {
            assert_eq!(term, 5);
        }
    }

    #[test]
    fn leaders_sorted_lexicographic() {
        let hbs = vec![hb("zoo", 5, 100), hb("alpha", 5, 100)];
        if let LeaderVerdict::SplitBrain { leaders } = check(&hbs, 105, 30) {
            assert_eq!(leaders, vec!["alpha".to_string(), "zoo".to_string()]);
        }
    }

    #[test]
    fn at_max_silence_boundary_still_fresh() {
        let hbs = vec![hb("a", 5, 70)];
        let v = check(&hbs, 100, 30);
        assert!(matches!(v, LeaderVerdict::SingleLeader { .. }));
    }
}
