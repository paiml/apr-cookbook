//! # Contracts-Macros Lemma Reuse Count
//!
//! Count how many times each lemma is referenced across theorems.
//! Returns the histogram and the most-reused lemma. Useful for
//! identifying foundational lemmas that need extra rigor.
//!
//! Demonstrates the **CMM.45** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mathlib4 reuse analytics.
//!
//! Run with: cargo run --example contracts_macros_lemma_reuse_count
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum ReuseVerdict {
    Ok {
        counts: Vec<(String, u32)>,
        most_reused: String,
        total_references: u32,
    },
    NoLemmas,
}

pub fn count(theorem_uses: &[(&str, Vec<&str>)]) -> ReuseVerdict {
    if theorem_uses.is_empty() {
        return ReuseVerdict::NoLemmas;
    }
    let mut hist: BTreeMap<String, u32> = BTreeMap::new();
    let mut total = 0u32;
    for (_, lemmas) in theorem_uses {
        for l in lemmas {
            *hist.entry((*l).to_string()).or_insert(0) += 1;
            total += 1;
        }
    }
    if hist.is_empty() {
        return ReuseVerdict::NoLemmas;
    }
    let mut counts: Vec<(String, u32)> = hist.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let most_reused = counts[0].0.clone();
    ReuseVerdict::Ok {
        counts,
        most_reused,
        total_references: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_lemma_reuse_count")?;

    let uses = vec![
        ("th1", vec!["lemma_a", "lemma_b"]),
        ("th2", vec!["lemma_a"]),
        ("th3", vec!["lemma_a", "lemma_c"]),
    ];
    println!("typical: {:?}", count(&uses));
    println!("no theorems: {:?}", count(&[]));

    let no_lemmas = vec![("th1", vec![])];
    println!("no lemmas: {:?}", count(&no_lemmas));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn most_reused_returned() {
        let uses = vec![
            ("t1", vec!["a", "b"]),
            ("t2", vec!["a"]),
            ("t3", vec!["a", "c"]),
        ];
        let v = count(&uses);
        if let ReuseVerdict::Ok { most_reused, .. } = v {
            assert_eq!(most_reused, "a");
        }
    }

    #[test]
    fn total_references_correct() {
        let uses = vec![("t1", vec!["a", "b"]), ("t2", vec!["a"])];
        let v = count(&uses);
        if let ReuseVerdict::Ok {
            total_references, ..
        } = v
        {
            assert_eq!(total_references, 3);
        }
    }

    #[test]
    fn no_theorems() {
        assert_eq!(count(&[]), ReuseVerdict::NoLemmas);
    }

    #[test]
    fn theorems_without_lemmas() {
        let uses = vec![("t1", vec![]), ("t2", vec![])];
        assert_eq!(count(&uses), ReuseVerdict::NoLemmas);
    }

    #[test]
    fn counts_sorted_desc() {
        let uses = vec![("t1", vec!["a", "a", "b"])];
        let v = count(&uses);
        if let ReuseVerdict::Ok { counts, .. } = v {
            assert_eq!(counts[0].0, "a");
            assert_eq!(counts[0].1, 2);
        }
    }

    #[test]
    fn alphabetical_tiebreak() {
        let uses = vec![("t1", vec!["x", "a"])];
        let v = count(&uses);
        if let ReuseVerdict::Ok { most_reused, .. } = v {
            // Both used once; alphabetically "a" < "x".
            assert_eq!(most_reused, "a");
        }
    }

    #[test]
    fn unique_lemmas() {
        let uses = vec![("t1", vec!["a", "b", "c"])];
        let v = count(&uses);
        if let ReuseVerdict::Ok {
            counts,
            total_references,
            ..
        } = v
        {
            assert_eq!(counts.len(), 3);
            assert_eq!(total_references, 3);
        }
    }

    #[test]
    fn duplicate_in_one_theorem() {
        let uses = vec![("t1", vec!["a", "a"])];
        let v = count(&uses);
        if let ReuseVerdict::Ok { counts, .. } = v {
            assert_eq!(counts[0].1, 2);
        }
    }

    #[test]
    fn many_theorems() {
        let uses: Vec<(&str, Vec<&str>)> = (0..100).map(|_| ("t", vec!["popular"])).collect();
        let v = count(&uses);
        if let ReuseVerdict::Ok { counts, .. } = v {
            assert_eq!(counts[0].1, 100);
        }
    }

    #[test]
    fn deterministic() {
        let uses = vec![("t1", vec!["a", "b"])];
        let a = count(&uses);
        let b = count(&uses);
        assert_eq!(a, b);
    }
}
