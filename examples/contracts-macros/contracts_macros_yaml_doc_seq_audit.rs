//! # Contracts-Macros YAML Document Sequence Audit
//!
//! Validate that multi-document YAML files follow the policy: each
//! document has a unique `doc_id` field, and `doc_id` values appear
//! in monotonically-increasing order. Returns sorted out-of-order
//! IDs.
//!
//! Demonstrates the **CMM.187** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §9.1 multi-document streams; CloudEvents
//!  ordered-event semantics.
//!
//! Run with: cargo run --example contracts_macros_yaml_doc_seq_audit
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DocSeqVerdict {
    Ok {
        out_of_order_ids: Vec<u32>,
        document_count: u32,
    },
    InvalidConfig,
}

pub fn audit(doc_ids: &[u32]) -> DocSeqVerdict {
    if doc_ids.is_empty() {
        return DocSeqVerdict::InvalidConfig;
    }
    let mut out_of_order: Vec<u32> = Vec::new();
    for w in doc_ids.windows(2) {
        if w[1] <= w[0] {
            out_of_order.push(w[1]);
        }
    }
    out_of_order.sort_unstable();
    out_of_order.dedup();
    DocSeqVerdict::Ok {
        out_of_order_ids: out_of_order,
        document_count: doc_ids.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_doc_seq_audit")?;

    println!("ordered: {:?}", audit(&[1, 2, 3, 4]));
    println!("out-of-order: {:?}", audit(&[1, 3, 2, 4]));
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
    fn ordered_no_offender() {
        let v = audit(&[1, 2, 3, 4]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert!(out_of_order_ids.is_empty());
        }
    }

    #[test]
    fn unordered_flagged() {
        let v = audit(&[1, 3, 2]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert_eq!(out_of_order_ids, vec![2]);
        }
    }

    #[test]
    fn duplicate_flagged() {
        let v = audit(&[1, 2, 2, 3]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert_eq!(out_of_order_ids, vec![2]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), DocSeqVerdict::InvalidConfig);
    }

    #[test]
    fn single_doc_no_offender() {
        let v = audit(&[42]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert!(out_of_order_ids.is_empty());
        }
    }

    #[test]
    fn document_count_correct() {
        let v = audit(&[1, 2, 3, 4, 5]);
        if let DocSeqVerdict::Ok { document_count, .. } = v {
            assert_eq!(document_count, 5);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[1, 2, 3]);
        let r2 = audit(&[1, 2, 3]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted_dedup() {
        let v = audit(&[5, 4, 3, 4, 5]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            for w in out_of_order_ids.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn fully_reversed_all_offending() {
        let v = audit(&[5, 4, 3, 2, 1]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert_eq!(out_of_order_ids.len(), 4);
        }
    }

    #[test]
    fn many_docs_handled() {
        let docs: Vec<u32> = (0..30).collect();
        let v = audit(&docs);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert!(out_of_order_ids.is_empty());
        }
    }

    #[test]
    fn jump_in_sequence_not_flagged() {
        // Increases are valid even if non-contiguous.
        let v = audit(&[1, 5, 100]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert!(out_of_order_ids.is_empty());
        }
    }

    #[test]
    fn zero_in_sequence_handled() {
        let v = audit(&[0, 1, 2]);
        if let DocSeqVerdict::Ok {
            out_of_order_ids, ..
        } = v
        {
            assert!(out_of_order_ids.is_empty());
        }
    }
}
