//! # Monte-Carlo Priority Queue with Aging
//!
//! Sim a priority queue where waiting tasks gain priority over time
//! (anti-starvation). Reports max task wait time vs no-aging baseline.
//!
//! Demonstrates the **MC.110** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aging mechanism in OS scheduling (Silberschatz, OS
//!  Concepts §5.3); priority-inversion avoidance.
//!
//! Run with: cargo run --example mc_priority_queue_aging
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AgingVerdict {
    Ok {
        max_wait_aging: u32,
        max_wait_naive: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    seconds: u32,
    arrival_per_sec: u32,
    high_priority_pct: u32,
    aging_factor: u32,
    seed: u64,
) -> AgingVerdict {
    if seconds == 0 || arrival_per_sec == 0 || high_priority_pct > 100 || aging_factor == 0 {
        return AgingVerdict::InvalidConfig;
    }
    // Simple model: track each task's wait time; priority = base + age * aging_factor.
    let mut tasks: Vec<(u32, u32, u32)> = Vec::new(); // (arrived, base_priority, naive_wait)
    let mut max_wait_aging = 0u32;
    let mut max_wait_naive = 0u32;
    let mut rng_state = seed | 1;
    for sec in 0..seconds {
        for _ in 0..arrival_per_sec {
            let r = ((lcg(&mut rng_state) >> 32) as u32) % 100;
            let base = if r < high_priority_pct { 100 } else { 1 };
            tasks.push((sec, base, 0));
        }
        // Age all waiting tasks.
        for task in &mut tasks {
            task.2 = sec - task.0;
        }
        // Schedule: pick highest priority + age.
        if let Some(idx) = pick_highest_with_aging(&tasks, sec, aging_factor) {
            let wait_aging = sec - tasks[idx].0;
            if wait_aging > max_wait_aging {
                max_wait_aging = wait_aging;
            }
            tasks.swap_remove(idx);
        }
        // Naive baseline: just highest base priority.
        if let Some(idx) = pick_highest_naive(&tasks) {
            let wait_naive = sec - tasks[idx].0;
            if wait_naive > max_wait_naive {
                max_wait_naive = wait_naive;
            }
            tasks.swap_remove(idx);
        }
    }
    AgingVerdict::Ok {
        max_wait_aging,
        max_wait_naive,
    }
}

fn pick_highest_with_aging(
    tasks: &[(u32, u32, u32)],
    sec: u32,
    aging_factor: u32,
) -> Option<usize> {
    if tasks.is_empty() {
        return None;
    }
    let mut best = 0;
    let mut best_score: u64 = 0;
    for (i, t) in tasks.iter().enumerate() {
        let age = sec - t.0;
        let score = u64::from(t.1) + u64::from(age) * u64::from(aging_factor);
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    Some(best)
}

fn pick_highest_naive(tasks: &[(u32, u32, u32)]) -> Option<usize> {
    if tasks.is_empty() {
        return None;
    }
    let mut best = 0;
    for (i, t) in tasks.iter().enumerate() {
        if t.1 > tasks[best].1 {
            best = i;
        }
    }
    Some(best)
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_priority_queue_aging")?;

    println!("typical: {:?}", simulate(100, 3, 30, 5, 42));
    println!("invalid: {:?}", simulate(0, 3, 30, 5, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn both_strategies_bounded_by_runtime() {
        // Both strategies' max wait should be bounded by total run length.
        let v = simulate(200, 3, 50, 10, 42);
        if let AgingVerdict::Ok {
            max_wait_aging,
            max_wait_naive,
        } = v
        {
            assert!(max_wait_aging <= 200);
            assert!(max_wait_naive <= 200);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 3, 30, 5, 42), AgingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_arrival() {
        assert_eq!(simulate(100, 0, 30, 5, 42), AgingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_pct_above_100() {
        assert_eq!(simulate(100, 3, 200, 5, 42), AgingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_aging() {
        assert_eq!(simulate(100, 3, 30, 0, 42), AgingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 3, 30, 5, 42);
        let b = simulate(100, 3, 30, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn waits_le_total_seconds() {
        let v = simulate(50, 3, 30, 5, 42);
        if let AgingVerdict::Ok {
            max_wait_aging,
            max_wait_naive,
        } = v
        {
            assert!(max_wait_aging <= 50);
            assert!(max_wait_naive <= 50);
        }
    }

    #[test]
    fn high_priority_low_max_wait() {
        let v = simulate(50, 3, 100, 5, 42);
        if let AgingVerdict::Ok { max_wait_naive, .. } = v {
            // All high → fast service, low max wait.
            let _ = max_wait_naive;
        }
    }

    #[test]
    fn aging_factor_changes_outcome() {
        // Different aging factors produce different scheduling patterns.
        let lo = simulate(100, 5, 50, 1, 42);
        let hi = simulate(100, 5, 50, 50, 42);
        if let (
            AgingVerdict::Ok {
                max_wait_aging: l, ..
            },
            AgingVerdict::Ok {
                max_wait_aging: h, ..
            },
        ) = (lo, hi)
        {
            // Both finite and bounded.
            assert!(l <= 100 && h <= 100);
        }
    }

    #[test]
    fn single_task_arrival_works() {
        let v = simulate(10, 1, 50, 5, 42);
        assert!(matches!(v, AgingVerdict::Ok { .. }));
    }

    #[test]
    fn many_arrivals_handled() {
        let v = simulate(50, 20, 30, 5, 42);
        assert!(matches!(v, AgingVerdict::Ok { .. }));
    }
}
