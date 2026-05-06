//! # Bundle Streaming Parallel Unpack Planner
//!
//! Multi-tensor unpack: distribute decompression across N worker
//! threads. Picker ratios:
//!   tensor_count < 4 OR cpu_cores == 1 → SequentialOnly
//!   tensor_count < cpu_cores → OneToOne
//!   tensor_count >= cpu_cores × 4 → WorkStealing (tasks > workers)
//!   between → BoundedThreadPool(cpu_cores)
//!
//! Demonstrates the **BUNDLE.21** recipe for PMAT-148 (bundling round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: rayon work-stealing thread pool model.
//!
//! Run with: cargo run --example bundle_streaming_unpack
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackStrategy {
    SequentialOnly,
    OneToOne,
    BoundedThreadPool { workers: u32 },
    WorkStealing { workers: u32 },
}

#[derive(Debug, PartialEq)]
pub enum UnpackVerdict {
    Ok {
        strategy: UnpackStrategy,
        expected_speedup: f64,
    },
    InvalidTensorCount,
    InvalidCoreCount,
}

pub fn pick(tensor_count: u32, cpu_cores: u32) -> UnpackVerdict {
    if tensor_count == 0 {
        return UnpackVerdict::InvalidTensorCount;
    }
    if cpu_cores == 0 {
        return UnpackVerdict::InvalidCoreCount;
    }
    let strategy = if tensor_count < 4 || cpu_cores == 1 {
        UnpackStrategy::SequentialOnly
    } else if tensor_count < cpu_cores {
        UnpackStrategy::OneToOne
    } else if tensor_count >= cpu_cores * 4 {
        UnpackStrategy::WorkStealing { workers: cpu_cores }
    } else {
        UnpackStrategy::BoundedThreadPool { workers: cpu_cores }
    };
    let expected_speedup = match strategy {
        UnpackStrategy::SequentialOnly => 1.0,
        UnpackStrategy::OneToOne => f64::from(tensor_count.min(cpu_cores)),
        UnpackStrategy::BoundedThreadPool { workers } => f64::from(workers) * 0.85,
        UnpackStrategy::WorkStealing { workers } => f64::from(workers) * 0.95,
    };
    UnpackVerdict::Ok {
        strategy,
        expected_speedup,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_streaming_unpack")?;

    println!("3 tensors, 8 cores: {:?}", pick(3, 8));
    println!("4 tensors, 8 cores: {:?}", pick(4, 8));
    println!("100 tensors, 8 cores: {:?}", pick(100, 8));
    println!("1000 tensors, 1 core: {:?}", pick(1000, 1));
    println!("invalid 0 tensors: {:?}", pick(0, 8));
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
    fn small_count_sequential() {
        let v = pick(3, 8);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::SequentialOnly);
        }
    }

    #[test]
    fn medium_count_one_to_one() {
        let v = pick(4, 8);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::OneToOne);
        }
    }

    #[test]
    fn high_count_work_stealing() {
        let v = pick(100, 8);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::WorkStealing { workers: 8 });
        }
    }

    #[test]
    fn balanced_count_bounded_pool() {
        // tensors = cores × 2 (between OneToOne and WorkStealing).
        let v = pick(16, 8);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::BoundedThreadPool { workers: 8 });
        }
    }

    #[test]
    fn single_core_always_sequential() {
        let v = pick(1000, 1);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::SequentialOnly);
        }
    }

    #[test]
    fn invalid_zero_tensors() {
        assert_eq!(pick(0, 8), UnpackVerdict::InvalidTensorCount);
    }

    #[test]
    fn invalid_zero_cores() {
        assert_eq!(pick(100, 0), UnpackVerdict::InvalidCoreCount);
    }

    #[test]
    fn sequential_speedup_one() {
        let v = pick(3, 8);
        if let UnpackVerdict::Ok {
            expected_speedup, ..
        } = v
        {
            assert_eq!(expected_speedup, 1.0);
        }
    }

    #[test]
    fn work_stealing_highest_speedup() {
        let v_seq = pick(2, 8);
        let v_ws = pick(100, 8);
        if let (
            UnpackVerdict::Ok {
                expected_speedup: seq,
                ..
            },
            UnpackVerdict::Ok {
                expected_speedup: ws,
                ..
            },
        ) = (v_seq, v_ws)
        {
            assert!(ws > seq);
        }
    }

    #[test]
    fn work_stealing_at_4x_cores() {
        // tensors == 4 × cores → WorkStealing.
        let v = pick(32, 8);
        if let UnpackVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UnpackStrategy::WorkStealing { workers: 8 });
        }
    }
}
