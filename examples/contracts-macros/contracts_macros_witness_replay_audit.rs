//! # Contracts-Macros Witness Replay Audit
//!
//! Verify witness records can be replayed: each witness needs a
//! seed + non-zero proof_steps. Returns sorted unreplayable IDs.
//!
//! Demonstrates the **CMM.162** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: hypothesis-python `@reproduce_failure` decorator;
//!  PropTest reproducible-shrink discipline.
//!
//! Run with: cargo run --example contracts_macros_witness_replay_audit
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReplayVerdict {
    Ok {
        unreplayable_ids: Vec<String>,
        replayable_count: u32,
    },
    InvalidConfig,
}

/// Items: (id, has_seed, proof_steps).
pub fn audit(items: &[(&str, bool, u32)]) -> ReplayVerdict {
    if items.is_empty() {
        return ReplayVerdict::InvalidConfig;
    }
    let mut unreplayable: Vec<String> = items
        .iter()
        .filter(|(_, seed, steps)| !*seed || *steps == 0)
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    unreplayable.sort();
    let replayable = items.len() as u32 - unreplayable.len() as u32;
    ReplayVerdict::Ok {
        unreplayable_ids: unreplayable,
        replayable_count: replayable,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_replay_audit")?;

    let items = [("w1", true, 5), ("w2", false, 3), ("w3", true, 0)];
    println!("audit: {:?}", audit(&items));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_witness_replayable() {
        let v = audit(&[("w", true, 5)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert!(unreplayable_ids.is_empty());
        }
    }

    #[test]
    fn missing_seed_unreplayable() {
        let v = audit(&[("w", false, 5)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(unreplayable_ids, vec!["w".to_string()]);
        }
    }

    #[test]
    fn zero_steps_unreplayable() {
        let v = audit(&[("w", true, 0)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(unreplayable_ids, vec!["w".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), ReplayVerdict::InvalidConfig);
    }

    #[test]
    fn replayable_count_correct() {
        let v = audit(&[("a", true, 5), ("b", false, 3), ("c", true, 5)]);
        if let ReplayVerdict::Ok {
            replayable_count, ..
        } = v
        {
            assert_eq!(replayable_count, 2);
        }
    }

    #[test]
    fn unreplayable_sorted() {
        let v = audit(&[("zeta", false, 5), ("alpha", false, 5)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(
                unreplayable_ids,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("w", true, 5)]);
        let r2 = audit(&[("w", true, 5)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_witnesses_handled() {
        let items: Vec<(&str, bool, u32)> = (0..30).map(|_| ("w", false, 5)).collect();
        let v = audit(&items);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(unreplayable_ids.len(), 30);
        }
    }

    #[test]
    fn all_replayable_returns_empty() {
        let v = audit(&[("a", true, 5), ("b", true, 10)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert!(unreplayable_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = audit(&[("café", false, 5)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(unreplayable_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn both_invalid_still_one_offender_entry() {
        // Missing seed AND zero steps → single offender entry.
        let v = audit(&[("w", false, 0)]);
        if let ReplayVerdict::Ok {
            unreplayable_ids, ..
        } = v
        {
            assert_eq!(unreplayable_ids.len(), 1);
        }
    }
}
