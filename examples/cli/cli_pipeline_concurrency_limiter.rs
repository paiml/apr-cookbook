//! # apr pipeline --concurrency — Fan-Out Concurrency Limiter
//!
//! Stages with N parallel sub-tasks must cap concurrency to avoid
//! overwhelming downstream resources. Default cap = min(N, 8 × cpus).
//! Per-stage override allowed if explicit. This recipe builds the
//! cap calculator + saturation classifier.
//!
//! Demonstrates the **PIPE.5** recipe for PMAT-121 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPE-001 + Little's Law (queueing theory)
//!
//! Run with: cargo run --example cli_pipeline_concurrency_limiter
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MULTIPLIER: u32 = 8;

#[derive(Debug, PartialEq)]
pub enum CapVerdict {
    Ok { effective_cap: u32 },
    InvalidTaskCount,
    InvalidCpuCount,
    OperatorOverrideExceedsTaskCount { override_cap: u32, tasks: u32 },
}

pub fn calc_cap(num_tasks: u32, num_cpus: u32, operator_override: Option<u32>) -> CapVerdict {
    if num_tasks == 0 {
        return CapVerdict::InvalidTaskCount;
    }
    if num_cpus == 0 {
        return CapVerdict::InvalidCpuCount;
    }
    let default_cap = num_tasks.min(MULTIPLIER * num_cpus);
    if let Some(o) = operator_override {
        if o == 0 {
            return CapVerdict::Ok { effective_cap: 1 }; // clamp 0 → 1
        }
        if o > num_tasks {
            return CapVerdict::OperatorOverrideExceedsTaskCount {
                override_cap: o,
                tasks: num_tasks,
            };
        }
        return CapVerdict::Ok { effective_cap: o };
    }
    CapVerdict::Ok {
        effective_cap: default_cap,
    }
}

#[derive(Debug, PartialEq)]
pub enum SaturationTier {
    Idle,
    Underutilized,
    Healthy,
    Saturated,
    Backpressured,
}

pub fn saturation(in_flight: u32, cap: u32) -> SaturationTier {
    if cap == 0 {
        return SaturationTier::Idle;
    }
    let pct = f64::from(in_flight) / f64::from(cap);
    if pct == 0.0 {
        SaturationTier::Idle
    } else if pct < 0.3 {
        SaturationTier::Underutilized
    } else if pct < 0.85 {
        SaturationTier::Healthy
    } else if pct <= 1.0 {
        SaturationTier::Saturated
    } else {
        SaturationTier::Backpressured
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_concurrency_limiter")?;

    println!("calc(50, 4 cpus, no override): {:?}", calc_cap(50, 4, None));
    println!(
        "calc(20, 4 cpus, override=5): {:?}",
        calc_cap(20, 4, Some(5))
    );
    println!(
        "calc(10, 4 cpus, override=20): {:?}",
        calc_cap(10, 4, Some(20))
    );
    for inflight in [0u32, 2, 6, 9, 11] {
        println!("saturation({inflight}/10): {:?}", saturation(inflight, 10));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_cap_is_min_of_tasks_and_8x_cpus() {
        // 50 tasks, 4 CPUs → min(50, 32) = 32.
        let v = calc_cap(50, 4, None);
        assert_eq!(v, CapVerdict::Ok { effective_cap: 32 });
    }

    #[test]
    fn fewer_tasks_than_cpu_budget_uses_task_count() {
        // 5 tasks, 4 CPUs → min(5, 32) = 5.
        let v = calc_cap(5, 4, None);
        assert_eq!(v, CapVerdict::Ok { effective_cap: 5 });
    }

    #[test]
    fn operator_override_respected() {
        let v = calc_cap(20, 4, Some(7));
        assert_eq!(v, CapVerdict::Ok { effective_cap: 7 });
    }

    #[test]
    fn operator_override_clamped_to_task_count() {
        let v = calc_cap(10, 4, Some(20));
        assert!(matches!(
            v,
            CapVerdict::OperatorOverrideExceedsTaskCount { .. }
        ));
    }

    #[test]
    fn operator_override_zero_clamps_to_one() {
        let v = calc_cap(10, 4, Some(0));
        assert_eq!(v, CapVerdict::Ok { effective_cap: 1 });
    }

    #[test]
    fn zero_tasks_invalid() {
        assert_eq!(calc_cap(0, 4, None), CapVerdict::InvalidTaskCount);
    }

    #[test]
    fn zero_cpus_invalid() {
        assert_eq!(calc_cap(10, 0, None), CapVerdict::InvalidCpuCount);
    }

    #[test]
    fn saturation_tiers_classified() {
        assert_eq!(saturation(0, 10), SaturationTier::Idle);
        assert_eq!(saturation(2, 10), SaturationTier::Underutilized);
        assert_eq!(saturation(6, 10), SaturationTier::Healthy);
        assert_eq!(saturation(9, 10), SaturationTier::Saturated);
        assert_eq!(saturation(11, 10), SaturationTier::Backpressured);
    }

    #[test]
    fn saturation_zero_cap_idle() {
        assert_eq!(saturation(5, 0), SaturationTier::Idle);
    }
}
