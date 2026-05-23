//! # Monitoring Ingest Backpressure
//!
//! Metric ingest queue strategies when load exceeds capacity:
//!   < 50% full   → AcceptAll
//!   50-80%       → SampleHalf (drop every other; keeps trends)
//!   80-95%       → DropLowSeverity (only Critical/Warn pass)
//!   ≥ 95%        → ShedAll (emergency dump)
//!
//! Demonstrates the **MON.29** recipe for PMAT-147 (monitoring round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS ALB / Linkerd backpressure patterns.
//!
//! Run with: cargo run --example monitor_ingest_backpressure
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestPolicy {
    AcceptAll,
    SampleHalf,
    DropLowSeverity,
    ShedAll,
}

#[derive(Debug, PartialEq)]
pub enum BackpressureVerdict {
    Ok {
        policy: IngestPolicy,
        utilization_pct: u32,
    },
    InvalidQueue,
}

pub fn pick(queue_depth: u32, queue_capacity: u32) -> BackpressureVerdict {
    if queue_capacity == 0 {
        return BackpressureVerdict::InvalidQueue;
    }
    let utilization_pct = ((u64::from(queue_depth) * 100) / u64::from(queue_capacity)) as u32;
    let policy = if utilization_pct < 50 {
        IngestPolicy::AcceptAll
    } else if utilization_pct < 80 {
        IngestPolicy::SampleHalf
    } else if utilization_pct < 95 {
        IngestPolicy::DropLowSeverity
    } else {
        IngestPolicy::ShedAll
    };
    BackpressureVerdict::Ok {
        policy,
        utilization_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_ingest_backpressure")?;

    println!("under-utilized: {:?}", pick(100, 1000));
    println!("half-full: {:?}", pick(600, 1000));
    println!("near-full: {:?}", pick(900, 1000));
    println!("emergency: {:?}", pick(990, 1000));
    println!("invalid: {:?}", pick(0, 0));
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
    fn low_util_accepts_all() {
        let v = pick(100, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::AcceptAll);
        }
    }

    #[test]
    fn medium_util_samples() {
        let v = pick(600, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::SampleHalf);
        }
    }

    #[test]
    fn high_util_drops_low() {
        let v = pick(900, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::DropLowSeverity);
        }
    }

    #[test]
    fn emergency_sheds_all() {
        let v = pick(990, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::ShedAll);
        }
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(pick(0, 0), BackpressureVerdict::InvalidQueue);
    }

    #[test]
    fn util_50_starts_sampling() {
        let v = pick(500, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::SampleHalf);
        }
    }

    #[test]
    fn util_80_starts_dropping() {
        let v = pick(800, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::DropLowSeverity);
        }
    }

    #[test]
    fn util_95_starts_shedding() {
        let v = pick(950, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::ShedAll);
        }
    }

    #[test]
    fn empty_queue_accepts_all() {
        let v = pick(0, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::AcceptAll);
        }
    }

    #[test]
    fn full_queue_sheds() {
        let v = pick(1000, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::ShedAll);
        }
    }

    #[test]
    fn over_capacity_sheds() {
        let v = pick(1500, 1000);
        if let BackpressureVerdict::Ok { policy, .. } = v {
            assert_eq!(policy, IngestPolicy::ShedAll);
        }
    }
}
