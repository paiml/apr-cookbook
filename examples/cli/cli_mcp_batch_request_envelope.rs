//! # apr mcp --batch — Batch Request Envelope
//!
//! JSON-RPC 2.0 batches: array of request objects. Constraints: array
//! non-empty; each element has unique `id`; total size capped at 100
//! requests per batch (default). Notifications (no `id`) allowed but
//! shouldn't dominate. This recipe builds the validator.
//!
//! Demonstrates the **MCP.6** recipe for PMAT-120 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-001 + JSON-RPC 2.0 §6
//!
//! Run with: cargo run --example cli_mcp_batch_request_envelope
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

const MAX_BATCH_SIZE: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestId {
    Number(i64),
    String(String),
    None, // notification
}

#[derive(Debug, PartialEq)]
pub enum BatchVerdict {
    Ok,
    Empty,
    ExceedsBatchSize { size: usize, max: usize },
    DuplicateId { id: RequestId },
    AllNotifications, // every entry is a notification — no replies expected
}

pub fn validate(ids: &[RequestId]) -> BatchVerdict {
    if ids.is_empty() {
        return BatchVerdict::Empty;
    }
    if ids.len() > MAX_BATCH_SIZE {
        return BatchVerdict::ExceedsBatchSize {
            size: ids.len(),
            max: MAX_BATCH_SIZE,
        };
    }
    let mut seen_num: HashSet<i64> = HashSet::new();
    let mut seen_str: HashSet<&str> = HashSet::new();
    let mut all_notifications = true;
    for id in ids {
        match id {
            RequestId::Number(n) => {
                all_notifications = false;
                if !seen_num.insert(*n) {
                    return BatchVerdict::DuplicateId { id: id.clone() };
                }
            }
            RequestId::String(s) => {
                all_notifications = false;
                if !seen_str.insert(s.as_str()) {
                    return BatchVerdict::DuplicateId { id: id.clone() };
                }
            }
            RequestId::None => {}
        }
    }
    if all_notifications {
        return BatchVerdict::AllNotifications;
    }
    BatchVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_batch_request_envelope")?;

    let ok = vec![
        RequestId::Number(1),
        RequestId::String("op-2".into()),
        RequestId::None,
    ];
    let dup = vec![RequestId::Number(1), RequestId::Number(1)];
    let all_notif = vec![RequestId::None, RequestId::None];
    println!("ok:        {:?}", validate(&ok));
    println!("dup:       {:?}", validate(&dup));
    println!("all_notif: {:?}", validate(&all_notif));
    println!("empty:     {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_mixed_batch_passes() {
        let ids = vec![
            RequestId::Number(1),
            RequestId::String("op-2".into()),
            RequestId::None,
        ];
        assert_eq!(validate(&ids), BatchVerdict::Ok);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[]), BatchVerdict::Empty);
    }

    #[test]
    fn over_max_rejected() {
        let ids: Vec<RequestId> = (0..(MAX_BATCH_SIZE + 1) as i64)
            .map(RequestId::Number)
            .collect();
        let v = validate(&ids);
        assert!(matches!(v, BatchVerdict::ExceedsBatchSize { .. }));
    }

    #[test]
    fn duplicate_number_id_detected() {
        let ids = vec![RequestId::Number(1), RequestId::Number(1)];
        let v = validate(&ids);
        assert!(matches!(v, BatchVerdict::DuplicateId { .. }));
    }

    #[test]
    fn duplicate_string_id_detected() {
        let ids = vec![RequestId::String("a".into()), RequestId::String("a".into())];
        let v = validate(&ids);
        assert!(matches!(v, BatchVerdict::DuplicateId { .. }));
    }

    #[test]
    fn number_and_string_id_can_coexist() {
        // "1" and 1 are not duplicates because they're different types.
        let ids = vec![RequestId::Number(1), RequestId::String("1".into())];
        assert_eq!(validate(&ids), BatchVerdict::Ok);
    }

    #[test]
    fn multiple_notifications_alone_flagged() {
        let ids = vec![RequestId::None, RequestId::None];
        assert_eq!(validate(&ids), BatchVerdict::AllNotifications);
    }

    #[test]
    fn at_max_passes() {
        let ids: Vec<RequestId> = (0..MAX_BATCH_SIZE as i64).map(RequestId::Number).collect();
        assert_eq!(validate(&ids), BatchVerdict::Ok);
    }

    #[test]
    fn single_request_passes() {
        assert_eq!(validate(&[RequestId::Number(42)]), BatchVerdict::Ok);
    }
}
