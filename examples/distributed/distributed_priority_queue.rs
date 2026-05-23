//! # Distributed Message Priority Queue
//!
//! Mesh messages have priorities (Critical/High/Normal/Low). Fair
//! scheduler ratio:
//!   Critical: deliver immediately
//!   High: 1× weight
//!   Normal: 2× of high
//!   Low: 5× of normal
//!
//! When draining at fixed budget per round, pick K from each tier per
//! the weights.
//!
//! Demonstrates the **DIST.15** recipe for PMAT-145 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Linux CFS scheduler weights + RabbitMQ priority queues.
//!
//! Run with: cargo run --example distributed_priority_queue
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Critical,
    High,
    Normal,
    Low,
}

#[derive(Debug, PartialEq)]
pub enum DrainVerdict {
    Ok {
        drained_critical: u32,
        drained_high: u32,
        drained_normal: u32,
        drained_low: u32,
    },
    InvalidBudget,
    InvalidQueueDepth,
}

pub fn drain(
    critical_depth: u32,
    high_depth: u32,
    normal_depth: u32,
    low_depth: u32,
    budget: u32,
) -> DrainVerdict {
    if budget == 0 {
        return DrainVerdict::InvalidBudget;
    }
    // Critical drains first, no weight cap.
    let drained_critical = critical_depth.min(budget);
    let mut remaining = budget - drained_critical;
    if remaining == 0 {
        return DrainVerdict::Ok {
            drained_critical,
            drained_high: 0,
            drained_normal: 0,
            drained_low: 0,
        };
    }
    // Weights: high=2, normal=4, low=10. Total = 16.
    // (Doubled from text to keep integer math clean.)
    let high_share = (remaining * 2 / 16).max(1);
    let normal_share = remaining * 4 / 16;
    let low_share = remaining * 10 / 16;
    let drained_high = high_depth.min(high_share);
    remaining -= drained_high;
    let drained_normal = normal_depth.min(normal_share.min(remaining));
    remaining -= drained_normal;
    let drained_low = low_depth.min(low_share.min(remaining));
    DrainVerdict::Ok {
        drained_critical,
        drained_high,
        drained_normal,
        drained_low,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_priority_queue")?;

    println!("typical: {:?}", drain(2, 50, 100, 200, 100));
    println!("only critical: {:?}", drain(80, 0, 0, 0, 100));
    println!("no critical: {:?}", drain(0, 50, 100, 200, 80));
    println!("invalid budget: {:?}", drain(0, 0, 0, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drainer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn critical_drained_first() {
        let v = drain(5, 0, 0, 0, 10);
        if let DrainVerdict::Ok {
            drained_critical, ..
        } = v
        {
            assert_eq!(drained_critical, 5);
        }
    }

    #[test]
    fn critical_caps_at_budget() {
        let v = drain(100, 50, 100, 100, 30);
        if let DrainVerdict::Ok {
            drained_critical,
            drained_high,
            ..
        } = v
        {
            assert_eq!(drained_critical, 30);
            assert_eq!(drained_high, 0);
        }
    }

    #[test]
    fn budget_distributed_after_critical() {
        let v = drain(0, 50, 100, 200, 80);
        if let DrainVerdict::Ok {
            drained_critical,
            drained_high,
            drained_normal,
            drained_low,
        } = v
        {
            assert_eq!(drained_critical, 0);
            assert!(drained_high > 0);
            assert!(drained_normal > 0);
            assert!(drained_low > 0);
            // Total drained should be roughly equal to budget.
            assert!(drained_high + drained_normal + drained_low <= 80);
        }
    }

    #[test]
    fn low_priority_gets_largest_share() {
        // Plenty of all kinds; check ratio.
        let v = drain(0, 1000, 1000, 1000, 16);
        if let DrainVerdict::Ok {
            drained_high,
            drained_normal,
            drained_low,
            ..
        } = v
        {
            // high=2, normal=4, low=10.
            assert_eq!(drained_high, 2);
            assert_eq!(drained_normal, 4);
            assert_eq!(drained_low, 10);
        }
    }

    #[test]
    fn invalid_budget_rejected() {
        assert_eq!(drain(0, 0, 0, 0, 0), DrainVerdict::InvalidBudget);
    }

    #[test]
    fn empty_queues_drain_zero() {
        let v = drain(0, 0, 0, 0, 100);
        if let DrainVerdict::Ok {
            drained_critical,
            drained_high,
            drained_normal,
            drained_low,
        } = v
        {
            assert_eq!(drained_critical, 0);
            assert_eq!(drained_high, 0);
            assert_eq!(drained_normal, 0);
            assert_eq!(drained_low, 0);
        }
    }

    #[test]
    fn drain_capped_at_queue_depth() {
        // Only 2 high-priority msgs; weights might allow more but cap at 2.
        let v = drain(0, 2, 100, 100, 80);
        if let DrainVerdict::Ok { drained_high, .. } = v {
            assert!(drained_high <= 2);
        }
    }

    #[test]
    fn critical_does_not_count_against_other_shares() {
        let with_crit = drain(20, 100, 100, 100, 100);
        let without_crit = drain(0, 100, 100, 100, 80);
        if let (
            DrainVerdict::Ok {
                drained_low: l_with,
                ..
            },
            DrainVerdict::Ok {
                drained_low: l_without,
                ..
            },
        ) = (with_crit, without_crit)
        {
            // After 20 critical, 80 left → same effective budget for low.
            assert_eq!(l_with, l_without);
        }
    }

    #[test]
    fn high_priority_weight_smallest() {
        let v = drain(0, 1000, 1000, 1000, 16);
        if let DrainVerdict::Ok {
            drained_high,
            drained_normal,
            drained_low,
            ..
        } = v
        {
            assert!(drained_high < drained_normal);
            assert!(drained_normal < drained_low);
        }
    }

    #[test]
    fn small_budget_assigns_at_least_minimum() {
        // Budget of 1 → high gets at least 1 (minimum 1 share).
        let v = drain(0, 5, 5, 5, 1);
        if let DrainVerdict::Ok { drained_high, .. } = v {
            assert!(drained_high >= 1);
        }
    }
}
