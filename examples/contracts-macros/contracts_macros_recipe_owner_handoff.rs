//! # Contracts-Macros Recipe Owner Handoff
//!
//! Track ownership-transition events for a recipe. Returns the final
//! owner, total handoffs, and a chain of unique owners (in order
//! of first ownership).
//!
//! Demonstrates the **CMM.152** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub CODEOWNERS handoff conventions; ITIL change-mgmt
//!  ownership-transfer audit trail.
//!
//! Run with: cargo run --example contracts_macros_recipe_owner_handoff
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HandoffVerdict {
    Ok {
        final_owner: String,
        handoff_count: u32,
        unique_owners_chain: Vec<String>,
    },
    InvalidConfig,
}

pub fn track(events: &[&str]) -> HandoffVerdict {
    if events.is_empty() {
        return HandoffVerdict::InvalidConfig;
    }
    let mut chain: Vec<String> = Vec::new();
    let mut handoff_count = 0u32;
    let mut last: Option<&&str> = None;
    for owner in events {
        if let Some(prev) = last {
            if prev != owner {
                handoff_count += 1;
            }
        }
        if !chain.iter().any(|o| o == owner) {
            chain.push((*owner).to_string());
        }
        last = Some(owner);
    }
    let final_owner = (*events.last().unwrap()).to_string();
    HandoffVerdict::Ok {
        final_owner,
        handoff_count,
        unique_owners_chain: chain,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_owner_handoff")?;

    let events = ["alice", "alice", "bob", "carol", "bob"];
    println!("track: {:?}", track(&events));
    println!("invalid: {:?}", track(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_owner_no_handoff() {
        let v = track(&["alice"]);
        if let HandoffVerdict::Ok { handoff_count, .. } = v {
            assert_eq!(handoff_count, 0);
        }
    }

    #[test]
    fn handoff_counted() {
        let v = track(&["alice", "bob"]);
        if let HandoffVerdict::Ok { handoff_count, .. } = v {
            assert_eq!(handoff_count, 1);
        }
    }

    #[test]
    fn final_owner_correct() {
        let v = track(&["a", "b", "c"]);
        if let HandoffVerdict::Ok { final_owner, .. } = v {
            assert_eq!(final_owner, "c");
        }
    }

    #[test]
    fn chain_preserves_order_first_seen() {
        let v = track(&["alice", "bob", "alice", "carol"]);
        if let HandoffVerdict::Ok {
            unique_owners_chain,
            ..
        } = v
        {
            assert_eq!(
                unique_owners_chain,
                vec!["alice".to_string(), "bob".to_string(), "carol".to_string()]
            );
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(track(&[]), HandoffVerdict::InvalidConfig);
    }

    #[test]
    fn repeats_no_handoff() {
        let v = track(&["alice", "alice", "alice"]);
        if let HandoffVerdict::Ok { handoff_count, .. } = v {
            assert_eq!(handoff_count, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = track(&["a", "b"]);
        let r2 = track(&["a", "b"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn returning_owner_handoff_counts() {
        // alice → bob → alice → 2 handoffs
        let v = track(&["alice", "bob", "alice"]);
        if let HandoffVerdict::Ok { handoff_count, .. } = v {
            assert_eq!(handoff_count, 2);
        }
    }

    #[test]
    fn many_handoffs_handled() {
        let mut events: Vec<&str> = Vec::new();
        for _ in 0..15 {
            events.push("a");
            events.push("b");
        }
        let v = track(&events);
        if let HandoffVerdict::Ok { handoff_count, .. } = v {
            assert_eq!(handoff_count, 29);
        }
    }

    #[test]
    fn unicode_owner_supported() {
        let v = track(&["café"]);
        if let HandoffVerdict::Ok { final_owner, .. } = v {
            assert_eq!(final_owner, "café");
        }
    }

    #[test]
    fn unique_chain_no_duplicates() {
        let v = track(&["a", "a", "b", "b", "c"]);
        if let HandoffVerdict::Ok {
            unique_owners_chain,
            ..
        } = v
        {
            assert_eq!(unique_owners_chain.len(), 3);
        }
    }
}
