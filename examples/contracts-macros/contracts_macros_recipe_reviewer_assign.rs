//! # Contracts-Macros Recipe Reviewer Assign
//!
//! Round-robin assign pending recipe reviews to a pool of reviewers,
//! skipping any reviewer the recipe author can't be reviewed by
//! (self-review). Returns sorted assignments and the rotation index.
//!
//! Demonstrates the **CMM.145** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub CODEOWNERS round-robin auto-assign; Gerrit's
//!  reviewer-suggester load balancing.
//!
//! Run with: cargo run --example contracts_macros_recipe_reviewer_assign
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReviewerVerdict {
    Ok {
        assignments: Vec<(String, String)>,
        next_rotation_idx: u32,
    },
    InvalidConfig,
}

pub fn assign(pending: &[(&str, &str)], reviewers: &[&str], start_idx: u32) -> ReviewerVerdict {
    if pending.is_empty() || reviewers.len() < 2 {
        return ReviewerVerdict::InvalidConfig;
    }
    let mut assignments: Vec<(String, String)> = Vec::with_capacity(pending.len());
    let mut idx = start_idx as usize;
    for (recipe, author) in pending {
        let mut tries = 0;
        let mut chosen = reviewers[idx % reviewers.len()];
        while chosen == *author && tries < reviewers.len() {
            idx += 1;
            chosen = reviewers[idx % reviewers.len()];
            tries += 1;
        }
        assignments.push(((*recipe).to_string(), chosen.to_string()));
        idx += 1;
    }
    assignments.sort();
    ReviewerVerdict::Ok {
        assignments,
        next_rotation_idx: (idx % reviewers.len()) as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_reviewer_assign")?;

    let pending = [("r1", "alice"), ("r2", "bob"), ("r3", "carol")];
    let reviewers = ["alice", "bob", "carol", "dave"];
    println!("assign: {:?}", assign(&pending, &reviewers, 0));
    println!("invalid: {:?}", assign(&[], &reviewers, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn each_pending_assigned() {
        let pending = [("r1", "alice"), ("r2", "bob")];
        let reviewers = ["alice", "bob", "carol"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments.len(), 2);
        }
    }

    #[test]
    fn author_never_self_assigned() {
        let pending = [("r", "alice")];
        let reviewers = ["alice", "bob"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_ne!(assignments[0].1, "alice");
        }
    }

    #[test]
    fn invalid_empty_pending() {
        let reviewers = ["a", "b"];
        assert_eq!(assign(&[], &reviewers, 0), ReviewerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_reviewers() {
        let pending = [("r", "a")];
        assert_eq!(assign(&pending, &["a"], 0), ReviewerVerdict::InvalidConfig);
    }

    #[test]
    fn assignments_sorted() {
        let pending = [("zeta", "a"), ("alpha", "b")];
        let reviewers = ["a", "b", "c"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments[0].0, "alpha");
            assert_eq!(assignments[1].0, "zeta");
        }
    }

    #[test]
    fn deterministic() {
        let pending = [("r", "x")];
        let reviewers = ["a", "b"];
        let r1 = assign(&pending, &reviewers, 0);
        let r2 = assign(&pending, &reviewers, 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rotation_idx_advances() {
        let pending = [("r1", "x"), ("r2", "y")];
        let reviewers = ["a", "b", "c"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok {
            next_rotation_idx, ..
        } = v
        {
            // Started at 0; assigned 2 → rotation_idx=2.
            assert_eq!(next_rotation_idx, 2);
        }
    }

    #[test]
    fn rotation_wraps_around() {
        let pending = [("r", "x")];
        let reviewers = ["a", "b"];
        let v = assign(&pending, &reviewers, 1);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments[0].1, "b");
        }
    }

    #[test]
    fn many_recipes_handled() {
        let pending: Vec<(&str, &str)> = (0..20).map(|_| ("r", "x")).collect();
        let reviewers = ["a", "b", "c"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments.len(), 20);
        }
    }

    #[test]
    fn unicode_names_supported() {
        let pending = [("r", "café")];
        let reviewers = ["café", "résumé"];
        let v = assign(&pending, &reviewers, 0);
        if let ReviewerVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments[0].1, "résumé");
        }
    }

    #[test]
    fn high_start_idx_handled() {
        let pending = [("r", "x")];
        let reviewers = ["a", "b"];
        let v = assign(&pending, &reviewers, 1_000_000);
        assert!(matches!(v, ReviewerVerdict::Ok { .. }));
    }
}
