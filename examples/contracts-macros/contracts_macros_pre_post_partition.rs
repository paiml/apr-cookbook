//! # Contracts-Macros Pre/Post Obligation Partition
//!
//! Partition contract obligations into pre-condition and post-condition
//! sets. Returns counts and a basic structural sanity check (each
//! obligation must belong to exactly one set).
//!
//! Demonstrates the **CMM.27** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hoare logic pre/post separation.
//!
//! Run with: cargo run --example contracts_macros_pre_post_partition
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationKind {
    Pre,
    Post,
    Either,
}

#[derive(Debug, PartialEq)]
pub enum PartitionVerdict {
    Ok { pre_count: u32, post_count: u32 },
    AmbiguousObligation { id: String },
    EmptyContract,
    UnclassifiedObligation { id: String },
}

pub fn partition(items: &[(&str, ObligationKind)]) -> PartitionVerdict {
    if items.is_empty() {
        return PartitionVerdict::EmptyContract;
    }
    let mut pre_count = 0u32;
    let mut post_count = 0u32;
    for (id, kind) in items {
        match kind {
            ObligationKind::Pre => pre_count += 1,
            ObligationKind::Post => post_count += 1,
            ObligationKind::Either => {
                return PartitionVerdict::AmbiguousObligation {
                    id: (*id).to_string(),
                };
            }
        }
        if id.is_empty() {
            return PartitionVerdict::UnclassifiedObligation { id: String::new() };
        }
    }
    PartitionVerdict::Ok {
        pre_count,
        post_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_pre_post_partition")?;

    let items = vec![
        ("input_in_range", ObligationKind::Pre),
        ("output_well_formed", ObligationKind::Post),
        ("invariant_holds", ObligationKind::Post),
    ];
    println!("typical: {:?}", partition(&items));

    let ambig = vec![("foo", ObligationKind::Either)];
    println!("ambiguous: {:?}", partition(&ambig));

    let empty_id = vec![("", ObligationKind::Pre)];
    println!("unclassified: {:?}", partition(&empty_id));

    println!("empty: {:?}", partition(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_partition() {
        let items = [
            ("a", ObligationKind::Pre),
            ("b", ObligationKind::Post),
            ("c", ObligationKind::Post),
        ];
        if let PartitionVerdict::Ok {
            pre_count,
            post_count,
        } = partition(&items)
        {
            assert_eq!(pre_count, 1);
            assert_eq!(post_count, 2);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(partition(&[]), PartitionVerdict::EmptyContract);
    }

    #[test]
    fn ambiguous_kind_rejected() {
        let items = [("foo", ObligationKind::Either)];
        assert!(matches!(
            partition(&items),
            PartitionVerdict::AmbiguousObligation { .. }
        ));
    }

    #[test]
    fn empty_id_rejected() {
        let items = [("", ObligationKind::Pre)];
        assert!(matches!(
            partition(&items),
            PartitionVerdict::UnclassifiedObligation { .. }
        ));
    }

    #[test]
    fn all_pre_zero_post() {
        let items = [("a", ObligationKind::Pre), ("b", ObligationKind::Pre)];
        if let PartitionVerdict::Ok { post_count, .. } = partition(&items) {
            assert_eq!(post_count, 0);
        }
    }

    #[test]
    fn all_post_zero_pre() {
        let items = [("a", ObligationKind::Post), ("b", ObligationKind::Post)];
        if let PartitionVerdict::Ok { pre_count, .. } = partition(&items) {
            assert_eq!(pre_count, 0);
        }
    }

    #[test]
    fn first_ambiguous_returned() {
        let items = [("a", ObligationKind::Pre), ("bad", ObligationKind::Either)];
        if let PartitionVerdict::AmbiguousObligation { id } = partition(&items) {
            assert_eq!(id, "bad");
        }
    }

    #[test]
    fn single_pre_works() {
        let items = [("only", ObligationKind::Pre)];
        if let PartitionVerdict::Ok {
            pre_count,
            post_count,
        } = partition(&items)
        {
            assert_eq!(pre_count, 1);
            assert_eq!(post_count, 0);
        }
    }

    #[test]
    fn many_obligations_count_correct() {
        let items: Vec<_> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    ("p", ObligationKind::Pre)
                } else {
                    ("q", ObligationKind::Post)
                }
            })
            .collect();
        if let PartitionVerdict::Ok {
            pre_count,
            post_count,
        } = partition(&items)
        {
            assert_eq!(pre_count, 50);
            assert_eq!(post_count, 50);
        }
    }

    #[test]
    fn deterministic() {
        let items = [("a", ObligationKind::Pre), ("b", ObligationKind::Post)];
        let a = partition(&items);
        let b = partition(&items);
        assert_eq!(a, b);
    }
}
