//! # Monte-Carlo Clock Drift Compensation
//!
//! Sim N distributed nodes with independent clock drift. Each second,
//! each node's clock advances by `1 + drift_ppm/1_000_000` actual
//! seconds. Reports max drift across all nodes after T seconds.
//!
//! Demonstrates the **MC.102** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NTP RFC 5905 §6 (drift estimation); Mills, "Computer
//!  Network Time Synchronization" (2010).
//!
//! Run with: cargo run --example mc_clock_drift_compensation
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Ok {
        max_drift_ms: u32,
        drifts_per_node: Vec<i32>,
    },
    InvalidConfig,
}

pub fn simulate(nodes: u32, seconds: u32, avg_drift_ppm: u32, seed: u64) -> DriftVerdict {
    if nodes == 0 || seconds == 0 || avg_drift_ppm == 0 {
        return DriftVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut drifts: Vec<i32> = Vec::with_capacity(nodes as usize);
    for _ in 0..nodes {
        // Per-node drift: ±avg_drift_ppm.
        let r = (lcg(&mut rng_state) >> 32) as i64;
        let signed_drift_ppm = (r % (2 * i64::from(avg_drift_ppm) + 1)) - i64::from(avg_drift_ppm);
        // Total drift over `seconds` = seconds × ppm × 1ms / 1000ppm.
        let drift_ms = signed_drift_ppm * i64::from(seconds) / 1000;
        drifts.push(drift_ms as i32);
    }
    let max_drift_ms = drifts.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0);
    DriftVerdict::Ok {
        max_drift_ms,
        drifts_per_node: drifts,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_clock_drift_compensation")?;

    println!("typical: {:?}", simulate(8, 3600, 50, 42));
    println!("high drift: {:?}", simulate(8, 3600, 1000, 42));
    println!("invalid: {:?}", simulate(0, 3600, 50, 42));
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
    fn invalid_zero_nodes() {
        assert_eq!(simulate(0, 3600, 50, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(8, 0, 50, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_drift() {
        assert_eq!(simulate(8, 3600, 0, 42), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(8, 3600, 50, 42);
        let b = simulate(8, 3600, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_drift_higher_max() {
        let lo = simulate(8, 3600, 50, 42);
        let hi = simulate(8, 3600, 5000, 42);
        if let (
            DriftVerdict::Ok {
                max_drift_ms: l, ..
            },
            DriftVerdict::Ok {
                max_drift_ms: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn longer_window_higher_drift() {
        let short = simulate(8, 60, 1000, 42);
        let long = simulate(8, 3600, 1000, 42);
        if let (
            DriftVerdict::Ok {
                max_drift_ms: s, ..
            },
            DriftVerdict::Ok {
                max_drift_ms: l, ..
            },
        ) = (short, long)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn drifts_count_matches_nodes() {
        let v = simulate(16, 3600, 50, 42);
        if let DriftVerdict::Ok {
            drifts_per_node, ..
        } = v
        {
            assert_eq!(drifts_per_node.len(), 16);
        }
    }

    #[test]
    fn max_drift_ge_individual_abs() {
        let v = simulate(8, 3600, 50, 42);
        if let DriftVerdict::Ok {
            max_drift_ms,
            drifts_per_node,
        } = v
        {
            for d in &drifts_per_node {
                assert!(d.unsigned_abs() <= max_drift_ms);
            }
        }
    }

    #[test]
    fn single_node_works() {
        let v = simulate(1, 3600, 50, 42);
        if let DriftVerdict::Ok {
            drifts_per_node, ..
        } = v
        {
            assert_eq!(drifts_per_node.len(), 1);
        }
    }

    #[test]
    fn realistic_drift_in_bounds() {
        // 50ppm × 3600s = 180_000 ppm-seconds = 180ms.
        let v = simulate(8, 3600, 50, 42);
        if let DriftVerdict::Ok { max_drift_ms, .. } = v {
            assert!(max_drift_ms < 1000);
        }
    }

    #[test]
    fn many_nodes_handled() {
        let v = simulate(64, 3600, 50, 42);
        assert!(matches!(v, DriftVerdict::Ok { .. }));
    }
}
