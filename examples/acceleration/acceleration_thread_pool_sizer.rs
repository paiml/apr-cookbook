//! # Acceleration Thread Pool Sizer
//!
//! Pool size depends on workload: compute-bound = num_cpus; I/O-bound
//! = 2-4 × num_cpus (latency hiding); mixed = 1.5 × num_cpus. Cap at
//! 64 (kernel scheduling overhead). This recipe builds the picker.
//!
//! Demonstrates the **ACCEL.6** recipe for PMAT-126 (acceleration coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goetz, B. (2006). Java Concurrency in Practice §8.2.
//!
//! Run with: cargo run --example acceleration_thread_pool_sizer
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    ComputeBound,
    IoBound,
    Mixed,
}

#[derive(Debug, PartialEq)]
pub enum SizerVerdict {
    Ok { pool_size: u32 },
    NoCpus,
}

const HARD_CAP: u32 = 64;

pub fn pick_pool_size(num_cpus: u32, kind: WorkloadKind) -> SizerVerdict {
    if num_cpus == 0 {
        return SizerVerdict::NoCpus;
    }
    let multiplier = match kind {
        WorkloadKind::ComputeBound => 1.0,
        WorkloadKind::IoBound => 3.0,
        WorkloadKind::Mixed => 1.5,
    };
    let raw = (f64::from(num_cpus) * multiplier).ceil() as u32;
    SizerVerdict::Ok {
        pool_size: raw.clamp(1, HARD_CAP),
    }
}

pub fn estimated_overhead_pct(pool_size: u32, num_cpus: u32) -> Option<f64> {
    if num_cpus == 0 {
        return None;
    }
    let ratio = f64::from(pool_size) / f64::from(num_cpus);
    if ratio <= 1.0 {
        Some(0.0)
    } else {
        // Heuristic: 5% per 1× oversubscription past 1×.
        Some((ratio - 1.0) * 5.0)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("acceleration_thread_pool_sizer")?;

    for cpus in [4u32, 8, 16, 32, 0] {
        for kind in [
            WorkloadKind::ComputeBound,
            WorkloadKind::IoBound,
            WorkloadKind::Mixed,
        ] {
            println!("cpus={cpus} {kind:?}  →  {:?}", pick_pool_size(cpus, kind));
        }
    }
    println!("overhead(16, 8): {:?}", estimated_overhead_pct(16, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn compute_bound_matches_cpu_count() {
        assert_eq!(
            pick_pool_size(8, WorkloadKind::ComputeBound),
            SizerVerdict::Ok { pool_size: 8 }
        );
    }

    #[test]
    fn io_bound_oversubscribes() {
        // I/O-bound = 3× cpus.
        assert_eq!(
            pick_pool_size(8, WorkloadKind::IoBound),
            SizerVerdict::Ok { pool_size: 24 }
        );
    }

    #[test]
    fn mixed_uses_1_5x() {
        // 8 × 1.5 = 12.
        assert_eq!(
            pick_pool_size(8, WorkloadKind::Mixed),
            SizerVerdict::Ok { pool_size: 12 }
        );
    }

    #[test]
    fn caps_at_hard_limit() {
        // 32 × 3 = 96 → capped to 64.
        assert_eq!(
            pick_pool_size(32, WorkloadKind::IoBound),
            SizerVerdict::Ok {
                pool_size: HARD_CAP
            }
        );
    }

    #[test]
    fn zero_cpus_rejected() {
        assert_eq!(
            pick_pool_size(0, WorkloadKind::ComputeBound),
            SizerVerdict::NoCpus
        );
    }

    #[test]
    fn one_cpu_handled() {
        assert_eq!(
            pick_pool_size(1, WorkloadKind::ComputeBound),
            SizerVerdict::Ok { pool_size: 1 }
        );
    }

    #[test]
    fn overhead_zero_at_or_below_cpus() {
        assert_eq!(estimated_overhead_pct(8, 8), Some(0.0));
        assert_eq!(estimated_overhead_pct(4, 8), Some(0.0));
    }

    #[test]
    fn overhead_grows_with_oversubscription() {
        // 2× oversubscription → 5%.
        let oh = estimated_overhead_pct(16, 8).unwrap();
        assert!((oh - 5.0).abs() < 1e-9);
    }

    #[test]
    fn overhead_zero_cpus_invalid() {
        assert!(estimated_overhead_pct(8, 0).is_none());
    }
}
