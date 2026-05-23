//! # Advanced KV-Cache Eviction Policy
//!
//! Long chat sessions exceed KV cache budget. Eviction policies:
//!   Lru: drop oldest accessed
//!   Fifo: drop oldest inserted
//!   SlidingWindow: keep only last K tokens
//!   Priority: keep system + recent user; drop assistant
//!
//! Picker maps (session_kind, budget_pressure) → policy.
//!
//! Demonstrates the **ADV.9** recipe for PMAT-141 (advanced round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vLLM PagedAttention prefix-cache eviction strategy.
//!
//! Run with: cargo run --example adv_kv_cache_eviction
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    SingleTurn,
    LongChat,
    Streaming,
    Batched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    Low,
    Moderate,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Lru,
    Fifo,
    SlidingWindow,
    Priority,
}

#[derive(Debug, PartialEq)]
pub enum EvictVerdict {
    Ok { policy: Policy, target_size: u32 },
    InvalidBudget,
}

pub fn pick(
    kind: SessionKind,
    pressure: Pressure,
    current_kib: u32,
    budget_kib: u32,
) -> EvictVerdict {
    if budget_kib == 0 {
        return EvictVerdict::InvalidBudget;
    }
    let policy = match (kind, pressure) {
        (SessionKind::SingleTurn, _) => Policy::Fifo,
        (SessionKind::LongChat, Pressure::Low) => Policy::Lru,
        (SessionKind::LongChat, Pressure::Moderate | Pressure::High) => Policy::Priority,
        (SessionKind::Streaming, _) => Policy::SlidingWindow,
        (SessionKind::Batched, _) => Policy::Fifo,
    };
    let target_size = match pressure {
        Pressure::Low => current_kib.min(budget_kib),
        Pressure::Moderate => budget_kib * 9 / 10,
        Pressure::High => budget_kib / 2,
    };
    EvictVerdict::Ok {
        policy,
        target_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_kv_cache_eviction")?;

    println!(
        "long chat low: {:?}",
        pick(SessionKind::LongChat, Pressure::Low, 1000, 4000)
    );
    println!(
        "long chat high: {:?}",
        pick(SessionKind::LongChat, Pressure::High, 4000, 4000)
    );
    println!(
        "streaming: {:?}",
        pick(SessionKind::Streaming, Pressure::Moderate, 2000, 4000)
    );
    println!(
        "single turn: {:?}",
        pick(SessionKind::SingleTurn, Pressure::Low, 100, 4000)
    );
    println!(
        "invalid: {:?}",
        pick(SessionKind::LongChat, Pressure::Low, 100, 0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn long_chat_low_pressure_uses_lru() {
        let v = pick(SessionKind::LongChat, Pressure::Low, 1000, 4000);
        if let EvictVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, Policy::Lru);
        }
    }

    #[test]
    fn long_chat_high_pressure_uses_priority() {
        let v = pick(SessionKind::LongChat, Pressure::High, 4000, 4000);
        if let EvictVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, Policy::Priority);
        }
    }

    #[test]
    fn streaming_uses_sliding_window() {
        let v = pick(SessionKind::Streaming, Pressure::Moderate, 2000, 4000);
        if let EvictVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, Policy::SlidingWindow);
        }
    }

    #[test]
    fn single_turn_uses_fifo() {
        let v = pick(SessionKind::SingleTurn, Pressure::Low, 100, 4000);
        if let EvictVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, Policy::Fifo);
        }
    }

    #[test]
    fn batched_uses_fifo() {
        let v = pick(SessionKind::Batched, Pressure::Low, 1000, 4000);
        if let EvictVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, Policy::Fifo);
        }
    }

    #[test]
    fn high_pressure_targets_half_budget() {
        let v = pick(SessionKind::LongChat, Pressure::High, 4000, 4000);
        if let EvictVerdict::Ok { target_size, .. } = v {
            assert_eq!(target_size, 2000);
        }
    }

    #[test]
    fn moderate_pressure_targets_90_percent() {
        let v = pick(SessionKind::LongChat, Pressure::Moderate, 4000, 4000);
        if let EvictVerdict::Ok { target_size, .. } = v {
            assert_eq!(target_size, 3600);
        }
    }

    #[test]
    fn low_pressure_keeps_current_within_budget() {
        let v = pick(SessionKind::LongChat, Pressure::Low, 1000, 4000);
        if let EvictVerdict::Ok { target_size, .. } = v {
            assert_eq!(target_size, 1000);
        }
    }

    #[test]
    fn low_pressure_clamps_to_budget() {
        // Current already over budget; clamp to budget.
        let v = pick(SessionKind::LongChat, Pressure::Low, 6000, 4000);
        if let EvictVerdict::Ok { target_size, .. } = v {
            assert_eq!(target_size, 4000);
        }
    }

    #[test]
    fn zero_budget_invalid() {
        assert_eq!(
            pick(SessionKind::LongChat, Pressure::Low, 100, 0),
            EvictVerdict::InvalidBudget
        );
    }
}
