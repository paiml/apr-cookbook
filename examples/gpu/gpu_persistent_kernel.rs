//! # GPU Persistent Kernel Picker
//!
//! Single-launch kernel resident across many small inputs vs.
//! per-input launch:
//!   small input + many calls + low launch latency → Persistent
//!   large input + few calls → Standard launch
//!
//! Demonstrates the **GPU.36** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA persistent threads CUDA technique.
//!
//! Run with: cargo run --example gpu_persistent_kernel
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchStrategy {
    Standard,
    Persistent,
    GraphCapture,
}

#[derive(Debug, PartialEq)]
pub enum LaunchVerdict {
    Ok {
        strategy: LaunchStrategy,
        expected_overhead_pct: u32,
    },
    InvalidConfig,
}

pub fn pick(
    input_size_bytes: u64,
    expected_calls_per_sec: u32,
    launch_overhead_us: u32,
) -> LaunchVerdict {
    if input_size_bytes == 0 || expected_calls_per_sec == 0 || launch_overhead_us == 0 {
        return LaunchVerdict::InvalidConfig;
    }
    let strategy = if input_size_bytes < 1024 * 1024 && expected_calls_per_sec > 1000 {
        LaunchStrategy::Persistent
    } else if expected_calls_per_sec > 100 {
        LaunchStrategy::GraphCapture
    } else {
        LaunchStrategy::Standard
    };
    let overhead_pct = match strategy {
        LaunchStrategy::Standard => 5,
        LaunchStrategy::GraphCapture => 1,
        LaunchStrategy::Persistent => 0,
    };
    LaunchVerdict::Ok {
        strategy,
        expected_overhead_pct: overhead_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_persistent_kernel")?;

    println!("small + frequent: {:?}", pick(1024, 5000, 10));
    println!("medium frequent: {:?}", pick(10 * 1024 * 1024, 500, 10));
    println!("large rare: {:?}", pick(1024 * 1024 * 1024, 10, 10));
    println!("invalid: {:?}", pick(0, 100, 10));
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
    fn small_frequent_persistent() {
        let v = pick(1024, 5000, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::Persistent);
        }
    }

    #[test]
    fn medium_frequent_graph() {
        let v = pick(10 * 1024 * 1024, 500, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::GraphCapture);
        }
    }

    #[test]
    fn large_rare_standard() {
        let v = pick(1024 * 1024 * 1024, 10, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::Standard);
        }
    }

    #[test]
    fn invalid_zero_size() {
        assert_eq!(pick(0, 100, 10), LaunchVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_calls() {
        assert_eq!(pick(1024, 0, 10), LaunchVerdict::InvalidConfig);
    }

    #[test]
    fn persistent_zero_overhead() {
        let v = pick(1024, 5000, 10);
        if let LaunchVerdict::Ok {
            expected_overhead_pct,
            ..
        } = v
        {
            assert_eq!(expected_overhead_pct, 0);
        }
    }

    #[test]
    fn standard_highest_overhead() {
        let std = pick(1024 * 1024 * 1024, 10, 10);
        let pers = pick(1024, 5000, 10);
        if let (
            LaunchVerdict::Ok {
                expected_overhead_pct: s,
                ..
            },
            LaunchVerdict::Ok {
                expected_overhead_pct: p,
                ..
            },
        ) = (std, pers)
        {
            assert!(s > p);
        }
    }

    #[test]
    fn graph_capture_low_overhead() {
        let v = pick(10 * 1024 * 1024, 500, 10);
        if let LaunchVerdict::Ok {
            expected_overhead_pct,
            ..
        } = v
        {
            assert_eq!(expected_overhead_pct, 1);
        }
    }

    #[test]
    fn boundary_at_1mib_persistent() {
        // Just under 1 MiB, frequent.
        let v = pick(1024 * 1024 - 1, 5000, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::Persistent);
        }
    }

    #[test]
    fn boundary_at_1mib_graph() {
        // Exactly 1 MiB, frequent → graph (not persistent).
        let v = pick(1024 * 1024, 5000, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::GraphCapture);
        }
    }

    #[test]
    fn boundary_at_1k_qps_persistent_threshold() {
        // Exactly 1000 qps + small input → graph (rule is `> 1000`).
        let v = pick(1024, 1000, 10);
        if let LaunchVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LaunchStrategy::GraphCapture);
        }
    }
}
