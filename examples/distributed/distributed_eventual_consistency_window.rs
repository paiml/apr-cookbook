//! # Distributed Eventual-Consistency Window
//!
//! Anti-entropy gossip ensures replicas converge eventually. The
//! convergence-window heuristic for N-node ring with fanout F:
//!   t_converge ≈ log_F(N) × gossip_interval_ms
//!
//! Picker chooses gossip_interval_ms to hit a target convergence time.
//! Trade-off: faster gossip = lower window but higher network noise.
//!
//! Demonstrates the **DIST.12** recipe for PMAT-142 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Demers et al. (1987). Epidemic Algorithms for Replicated Database Maintenance.
//!
//! Run with: cargo run --example distributed_eventual_consistency_window
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WindowVerdict {
    Ok {
        gossip_interval_ms: u32,
        rounds_to_converge: u32,
        actual_convergence_ms: u32,
    },
    InvalidNodeCount,
    InvalidFanout,
    InvalidTarget,
    TargetUnreachable {
        min_possible_ms: u32,
    },
}

const MIN_GOSSIP_MS: u32 = 50;
const MAX_GOSSIP_MS: u32 = 60_000;

pub fn pick(n_nodes: u32, fanout: u32, target_convergence_ms: u32) -> WindowVerdict {
    if n_nodes < 2 {
        return WindowVerdict::InvalidNodeCount;
    }
    if fanout < 2 || fanout >= n_nodes {
        return WindowVerdict::InvalidFanout;
    }
    if target_convergence_ms == 0 {
        return WindowVerdict::InvalidTarget;
    }
    let rounds = (f64::from(n_nodes).log(f64::from(fanout))).ceil() as u32;
    let min_possible_ms = rounds * MIN_GOSSIP_MS;
    if target_convergence_ms < min_possible_ms {
        return WindowVerdict::TargetUnreachable { min_possible_ms };
    }
    let interval = (target_convergence_ms / rounds.max(1)).clamp(MIN_GOSSIP_MS, MAX_GOSSIP_MS);
    WindowVerdict::Ok {
        gossip_interval_ms: interval,
        rounds_to_converge: rounds,
        actual_convergence_ms: interval * rounds,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_eventual_consistency_window")?;

    println!("100 nodes, fanout 3, 5s: {:?}", pick(100, 3, 5_000));
    println!("1000 nodes, fanout 5, 30s: {:?}", pick(1000, 5, 30_000));
    println!("100 nodes target unreachable: {:?}", pick(100, 3, 100));
    println!("invalid n=1: {:?}", pick(1, 3, 1000));
    println!("invalid fanout=1: {:?}", pick(100, 1, 1000));
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
    fn typical_100_nodes() {
        // log_3(100) ≈ 4.19 → ceil 5 rounds. 5000/5 = 1000 ms interval.
        let v = pick(100, 3, 5_000);
        if let WindowVerdict::Ok {
            rounds_to_converge, ..
        } = v
        {
            assert_eq!(rounds_to_converge, 5);
        }
    }

    #[test]
    fn larger_fanout_fewer_rounds() {
        let v_low = pick(1000, 3, 30_000);
        let v_high = pick(1000, 10, 30_000);
        if let (
            WindowVerdict::Ok {
                rounds_to_converge: l,
                ..
            },
            WindowVerdict::Ok {
                rounds_to_converge: h,
                ..
            },
        ) = (v_low, v_high)
        {
            assert!(h < l);
        }
    }

    #[test]
    fn invalid_one_node_rejected() {
        assert_eq!(pick(1, 3, 1000), WindowVerdict::InvalidNodeCount);
    }

    #[test]
    fn invalid_fanout_one_rejected() {
        assert_eq!(pick(100, 1, 1000), WindowVerdict::InvalidFanout);
    }

    #[test]
    fn fanout_at_or_above_n_rejected() {
        assert_eq!(pick(100, 100, 1000), WindowVerdict::InvalidFanout);
        assert_eq!(pick(100, 200, 1000), WindowVerdict::InvalidFanout);
    }

    #[test]
    fn zero_target_rejected() {
        assert_eq!(pick(100, 3, 0), WindowVerdict::InvalidTarget);
    }

    #[test]
    fn target_too_aggressive_unreachable() {
        // 5 rounds × 50 ms = 250 ms minimum. Asking for 100 ms → unreachable.
        let v = pick(100, 3, 100);
        assert!(matches!(v, WindowVerdict::TargetUnreachable { .. }));
    }

    #[test]
    fn actual_convergence_within_target() {
        // For valid pick, actual_convergence_ms ≤ target_ms (by construction).
        if let WindowVerdict::Ok {
            actual_convergence_ms,
            ..
        } = pick(100, 3, 5000)
        {
            assert!(actual_convergence_ms <= 5000);
        }
    }

    #[test]
    fn interval_clamped_to_min() {
        // Tiny target → clamped at MIN_GOSSIP_MS.
        let v = pick(100, 3, 250);
        if let WindowVerdict::Ok {
            gossip_interval_ms, ..
        } = v
        {
            assert_eq!(gossip_interval_ms, MIN_GOSSIP_MS);
        }
    }

    #[test]
    fn many_nodes_more_rounds() {
        let small = pick(100, 3, 30_000);
        let large = pick(10_000, 3, 30_000);
        if let (
            WindowVerdict::Ok {
                rounds_to_converge: s,
                ..
            },
            WindowVerdict::Ok {
                rounds_to_converge: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }
}
