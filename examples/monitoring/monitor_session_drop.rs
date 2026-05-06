//! # Monitoring Streaming Session Drop Detector
//!
//! Detect aborted SSE/WebSocket streaming sessions: client disconnect
//! before stream end. Verdict based on session count vs completion
//! count and recent_dropped count.
//!
//! Demonstrates the **MON.48** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Server-Sent Events (RFC 6455 / WS) keep-alive recovery.
//!
//! Run with: cargo run --example monitor_session_drop
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropVerdict {
    Healthy { drop_pct: f64 },
    Elevated { drop_pct: f64 },
    Critical { drop_pct: f64 },
    NoSessions,
}

pub fn check(total_started: u64, total_dropped: u64) -> DropVerdict {
    if total_started == 0 {
        return DropVerdict::NoSessions;
    }
    let drops = total_dropped.min(total_started);
    let drop_pct = (drops as f64 / total_started as f64) * 100.0;
    if drop_pct >= 10.0 {
        DropVerdict::Critical { drop_pct }
    } else if drop_pct >= 2.0 {
        DropVerdict::Elevated { drop_pct }
    } else {
        DropVerdict::Healthy { drop_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_session_drop")?;

    println!("healthy: {:?}", check(10_000, 50));
    println!("elevated: {:?}", check(10_000, 500));
    println!("critical: {:?}", check(10_000, 1500));
    println!("no sessions: {:?}", check(0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthy_below_2_pct() {
        let v = check(10_000, 50);
        assert!(matches!(v, DropVerdict::Healthy { .. }));
    }

    #[test]
    fn elevated_2_to_10_pct() {
        let v = check(10_000, 500);
        assert!(matches!(v, DropVerdict::Elevated { .. }));
    }

    #[test]
    fn critical_above_10_pct() {
        let v = check(10_000, 1500);
        assert!(matches!(v, DropVerdict::Critical { .. }));
    }

    #[test]
    fn no_sessions_classified() {
        assert_eq!(check(0, 0), DropVerdict::NoSessions);
    }

    #[test]
    fn boundary_at_2_pct_elevated() {
        let v = check(10_000, 200);
        assert!(matches!(v, DropVerdict::Elevated { .. }));
    }

    #[test]
    fn boundary_at_10_pct_critical() {
        let v = check(10_000, 1000);
        assert!(matches!(v, DropVerdict::Critical { .. }));
    }

    #[test]
    fn drops_capped_at_started() {
        // drops > started: cap at 100%.
        let v = check(100, 1000);
        if let DropVerdict::Critical { drop_pct } = v {
            assert!((drop_pct - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_drops_healthy() {
        let v = check(1000, 0);
        if let DropVerdict::Healthy { drop_pct } = v {
            assert!((drop_pct - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn small_population_works() {
        let v = check(10, 1);
        // 10% → critical.
        assert!(matches!(v, DropVerdict::Critical { .. }));
    }

    #[test]
    fn drop_pct_value_correct() {
        let v = check(1000, 30);
        if let DropVerdict::Elevated { drop_pct } = v {
            assert!((drop_pct - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(10_000, 500);
        let b = check(10_000, 500);
        assert_eq!(a, b);
    }
}
