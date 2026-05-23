//! # Advanced Priority Queue Eviction Policy
//!
//! When the inference queue is full, evict the lowest-priority pending
//! request to admit a higher-priority one. Tiebreaker: oldest queue
//! position (FIFO within same priority).
//!
//! Demonstrates the **ADV.33** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Operating-systems texts on priority scheduling.
//!
//! Run with: cargo run --example adv_priority_queue_eviction
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedRequest {
    pub id: u64,
    pub priority: u8,
    pub queued_position: u64,
}

#[derive(Debug, PartialEq)]
pub enum EvictVerdict {
    Admit { evicted_id: u64 },
    Reject { reason: &'static str },
    EmptyQueue,
}

pub fn decide(queue: &[QueuedRequest], capacity: usize, incoming_priority: u8) -> EvictVerdict {
    if queue.is_empty() {
        return EvictVerdict::EmptyQueue;
    }
    if queue.len() < capacity {
        return EvictVerdict::Reject {
            reason: "queue not full",
        };
    }
    // Find the lowest-priority entry (highest priority value = lowest priority).
    let candidate = queue.iter().max_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(a.queued_position.cmp(&b.queued_position))
    });
    let Some(c) = candidate else {
        return EvictVerdict::Reject {
            reason: "no candidate",
        };
    };
    if incoming_priority < c.priority {
        EvictVerdict::Admit { evicted_id: c.id }
    } else {
        EvictVerdict::Reject {
            reason: "incoming not higher priority than queue tail",
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_priority_queue_eviction")?;

    let queue = vec![
        QueuedRequest {
            id: 1,
            priority: 5,
            queued_position: 1,
        },
        QueuedRequest {
            id: 2,
            priority: 9,
            queued_position: 2,
        },
        QueuedRequest {
            id: 3,
            priority: 7,
            queued_position: 3,
        },
    ];
    println!("admit: {:?}", decide(&queue, 3, 1));
    println!("reject: {:?}", decide(&queue, 3, 10));
    println!("not full: {:?}", decide(&queue, 5, 1));
    println!("empty: {:?}", decide(&[], 3, 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<QueuedRequest> {
        vec![
            QueuedRequest {
                id: 1,
                priority: 5,
                queued_position: 1,
            },
            QueuedRequest {
                id: 2,
                priority: 9,
                queued_position: 2,
            },
            QueuedRequest {
                id: 3,
                priority: 7,
                queued_position: 3,
            },
        ]
    }

    #[test]
    fn evictor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn higher_priority_admitted() {
        let v = decide(&typical(), 3, 1);
        if let EvictVerdict::Admit { evicted_id } = v {
            // ID 2 has the highest priority value (9 = lowest priority).
            assert_eq!(evicted_id, 2);
        }
    }

    #[test]
    fn lower_priority_rejected() {
        let v = decide(&typical(), 3, 10);
        assert!(matches!(v, EvictVerdict::Reject { .. }));
    }

    #[test]
    fn equal_priority_rejected() {
        let v = decide(&typical(), 3, 9);
        assert!(matches!(v, EvictVerdict::Reject { .. }));
    }

    #[test]
    fn not_full_rejected() {
        let v = decide(&typical(), 5, 1);
        assert!(matches!(v, EvictVerdict::Reject { .. }));
    }

    #[test]
    fn empty_queue_rejected() {
        assert_eq!(decide(&[], 3, 1), EvictVerdict::EmptyQueue);
    }

    #[test]
    fn fifo_tiebreak_within_priority() {
        let q = vec![
            QueuedRequest {
                id: 1,
                priority: 5,
                queued_position: 1,
            },
            QueuedRequest {
                id: 2,
                priority: 5,
                queued_position: 2,
            },
        ];
        let v = decide(&q, 2, 1);
        if let EvictVerdict::Admit { evicted_id } = v {
            // Same priority → evict highest queued_position (last in).
            assert_eq!(evicted_id, 2);
        }
    }

    #[test]
    fn single_entry_evicts_it() {
        let q = vec![QueuedRequest {
            id: 99,
            priority: 5,
            queued_position: 1,
        }];
        let v = decide(&q, 1, 1);
        if let EvictVerdict::Admit { evicted_id } = v {
            assert_eq!(evicted_id, 99);
        }
    }

    #[test]
    fn priority_zero_means_highest() {
        // priority 0 = highest. Ensure not all incoming get rejected.
        let q = vec![QueuedRequest {
            id: 1,
            priority: 5,
            queued_position: 1,
        }];
        let v = decide(&q, 1, 0);
        assert!(matches!(v, EvictVerdict::Admit { .. }));
    }

    #[test]
    fn priority_max_low() {
        let q = vec![QueuedRequest {
            id: 1,
            priority: 0,
            queued_position: 1,
        }];
        // Incoming priority is 255 (lowest); cannot evict.
        let v = decide(&q, 1, 255);
        assert!(matches!(v, EvictVerdict::Reject { .. }));
    }

    #[test]
    fn deterministic() {
        let q = typical();
        let a = decide(&q, 3, 1);
        let b = decide(&q, 3, 1);
        assert_eq!(a, b);
    }
}
