//! # Advanced Continuous Batching Admission Decision
//!
//! vLLM-style continuous batching: incoming request decision is whether
//! to admit immediately (room in current batch), wait for next batch
//! (full batch but cheap to wait), or reject (overloaded).
//!
//! Decision rules:
//!   running_requests < max_batch & memory_ok → AdmitNow
//!   running_requests >= max_batch & queue_depth < max_queue → Queue
//!   queue_depth >= max_queue → Reject
//!
//! Demonstrates the **ADV.12** recipe for PMAT-142 (advanced round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vLLM continuous-batching design (Kwon et al. 2023).
//!
//! Run with: cargo run --example adv_continuous_batching
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AdmissionVerdict {
    AdmitNow { batch_after: u32 },
    QueueForNextBatch { queue_position: u32 },
    Reject { reason: &'static str },
    InvalidLimits,
}

pub fn decide(
    running_requests: u32,
    queue_depth: u32,
    max_batch_size: u32,
    max_queue_depth: u32,
    free_kv_cache_kib: u64,
    estimated_kv_kib: u64,
) -> AdmissionVerdict {
    if max_batch_size == 0 || max_queue_depth == 0 {
        return AdmissionVerdict::InvalidLimits;
    }
    if estimated_kv_kib == 0 {
        return AdmissionVerdict::InvalidLimits;
    }
    if running_requests < max_batch_size && free_kv_cache_kib >= estimated_kv_kib {
        return AdmissionVerdict::AdmitNow {
            batch_after: running_requests + 1,
        };
    }
    if queue_depth < max_queue_depth {
        return AdmissionVerdict::QueueForNextBatch {
            queue_position: queue_depth + 1,
        };
    }
    AdmissionVerdict::Reject {
        reason: "queue full",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_continuous_batching")?;

    println!("admit now: {:?}", decide(10, 0, 32, 100, 8 * 1024, 100));
    println!("queue: {:?}", decide(32, 50, 32, 100, 8 * 1024, 100));
    println!("reject: {:?}", decide(32, 100, 32, 100, 8 * 1024, 100));
    println!("kv shortage queues: {:?}", decide(10, 50, 32, 100, 50, 100));
    println!("invalid: {:?}", decide(0, 0, 0, 0, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batching_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn room_in_batch_admit_now() {
        let v = decide(10, 0, 32, 100, 8192, 100);
        assert!(matches!(v, AdmissionVerdict::AdmitNow { .. }));
    }

    #[test]
    fn full_batch_room_in_queue_queues() {
        let v = decide(32, 50, 32, 100, 8192, 100);
        assert!(matches!(v, AdmissionVerdict::QueueForNextBatch { .. }));
    }

    #[test]
    fn full_queue_rejects() {
        let v = decide(32, 100, 32, 100, 8192, 100);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }));
    }

    #[test]
    fn kv_shortage_queues_even_if_batch_room() {
        // Batch has room but KV cache doesn't fit one more.
        let v = decide(10, 50, 32, 100, 50, 100);
        assert!(matches!(v, AdmissionVerdict::QueueForNextBatch { .. }));
    }

    #[test]
    fn batch_position_after_admission() {
        if let AdmissionVerdict::AdmitNow { batch_after } = decide(10, 0, 32, 100, 8192, 100) {
            assert_eq!(batch_after, 11);
        }
    }

    #[test]
    fn queue_position_after_queueing() {
        if let AdmissionVerdict::QueueForNextBatch { queue_position } =
            decide(32, 5, 32, 100, 8192, 100)
        {
            assert_eq!(queue_position, 6);
        }
    }

    #[test]
    fn invalid_limits_zero_batch_rejected() {
        assert_eq!(
            decide(0, 0, 0, 100, 8192, 100),
            AdmissionVerdict::InvalidLimits
        );
    }

    #[test]
    fn invalid_zero_queue_rejected() {
        assert_eq!(
            decide(0, 0, 32, 0, 8192, 100),
            AdmissionVerdict::InvalidLimits
        );
    }

    #[test]
    fn invalid_zero_estimated_kv_rejected() {
        assert_eq!(
            decide(10, 0, 32, 100, 8192, 0),
            AdmissionVerdict::InvalidLimits
        );
    }

    #[test]
    fn batch_at_limit_queues_one_in() {
        // running == max → falls into queue path.
        let v = decide(32, 0, 32, 100, 8192, 100);
        assert!(matches!(v, AdmissionVerdict::QueueForNextBatch { .. }));
    }

    #[test]
    fn empty_batch_admits() {
        let v = decide(0, 0, 32, 100, 8192, 100);
        if let AdmissionVerdict::AdmitNow { batch_after } = v {
            assert_eq!(batch_after, 1);
        }
    }
}
