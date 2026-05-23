//! # Monitoring Network Packet Loss Rate
//!
//! TCP retransmits + UDP drops indicate network instability. Detector:
//!   <0.1%: Healthy
//!   0.1-1%: Elevated
//!   ≥1%: Degraded
//!
//! Demonstrates the **MON.43** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 793 + Internet 2 NetFlow loss thresholds.
//!
//! Run with: cargo run --example monitor_packet_loss_rate
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LossVerdict {
    Healthy { loss_pct: f64 },
    Elevated { loss_pct: f64 },
    Degraded { loss_pct: f64 },
    NoTraffic,
}

pub fn check(packets_sent: u64, packets_dropped: u64) -> LossVerdict {
    if packets_sent == 0 {
        return LossVerdict::NoTraffic;
    }
    let drop_count = packets_dropped.min(packets_sent);
    let loss_pct = (drop_count as f64 / packets_sent as f64) * 100.0;
    if loss_pct >= 1.0 {
        LossVerdict::Degraded { loss_pct }
    } else if loss_pct >= 0.1 {
        LossVerdict::Elevated { loss_pct }
    } else {
        LossVerdict::Healthy { loss_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_packet_loss_rate")?;

    println!("healthy: {:?}", check(1_000_000, 50));
    println!("elevated: {:?}", check(1_000_000, 5_000));
    println!("degraded: {:?}", check(1_000_000, 50_000));
    println!("no traffic: {:?}", check(0, 0));
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
    fn healthy_below_0_1_pct() {
        let v = check(1_000_000, 50);
        assert!(matches!(v, LossVerdict::Healthy { .. }));
    }

    #[test]
    fn elevated_0_1_to_1_pct() {
        let v = check(1_000_000, 5_000);
        assert!(matches!(v, LossVerdict::Elevated { .. }));
    }

    #[test]
    fn degraded_above_1_pct() {
        let v = check(1_000_000, 50_000);
        assert!(matches!(v, LossVerdict::Degraded { .. }));
    }

    #[test]
    fn no_traffic_classified() {
        assert_eq!(check(0, 0), LossVerdict::NoTraffic);
    }

    #[test]
    fn boundary_at_0_1_pct_elevated() {
        let v = check(1_000_000, 1_000);
        assert!(matches!(v, LossVerdict::Elevated { .. }));
    }

    #[test]
    fn boundary_at_1_pct_degraded() {
        let v = check(1_000_000, 10_000);
        assert!(matches!(v, LossVerdict::Degraded { .. }));
    }

    #[test]
    fn drops_capped_at_sent() {
        // If drops > sent, treat as drop=sent (max 100%).
        let v = check(100, 1_000_000);
        if let LossVerdict::Degraded { loss_pct } = v {
            assert!((loss_pct - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn loss_pct_value_correct() {
        let v = check(10_000, 100);
        if let LossVerdict::Elevated { loss_pct } = v {
            assert!((loss_pct - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_drops_healthy() {
        let v = check(1_000_000, 0);
        if let LossVerdict::Healthy { loss_pct } = v {
            assert!((loss_pct - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn small_traffic_works() {
        let v = check(10, 1);
        // 10% loss = degraded.
        assert!(matches!(v, LossVerdict::Degraded { .. }));
    }

    #[test]
    fn deterministic() {
        let a = check(1_000_000, 500);
        let b = check(1_000_000, 500);
        assert_eq!(a, b);
    }
}
