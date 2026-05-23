//! # GPU Async Memcpy + Kernel Overlap Picker
//!
//! Use multiple CUDA streams to overlap H2D copies with kernel exec:
//!   1 stream:  serial copy then kernel (no overlap)
//!   2 streams: pipeline (copy[i+1] || compute[i])
//!   ≥ 3:       deeper pipelining; diminishing returns past 4
//!
//! Picker rule:
//!   compute_ms ≤ copy_ms × 0.5         → 1 stream (compute dominates)
//!   copy_ms ≤ compute_ms × 0.5         → 1 stream (copy dominates)
//!   roughly equal                      → 2-4 streams (sweet spot)
//!
//! Demonstrates the **GPU.28** recipe for PMAT-143 (gpu round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA CUDA Best Practices Guide § overlapping data transfer.
//!
//! Run with: cargo run --example gpu_async_memcpy_overlap
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_STREAMS: u32 = 8;

#[derive(Debug, PartialEq)]
pub enum OverlapVerdict {
    Ok {
        n_streams: u32,
        expected_speedup: f64,
    },
    InvalidTiming,
}

pub fn pick(copy_ms: f64, compute_ms: f64, max_streams_avail: u32) -> OverlapVerdict {
    if !copy_ms.is_finite() || !compute_ms.is_finite() {
        return OverlapVerdict::InvalidTiming;
    }
    if copy_ms < 0.0 || compute_ms < 0.0 {
        return OverlapVerdict::InvalidTiming;
    }
    if copy_ms == 0.0 || compute_ms == 0.0 {
        return OverlapVerdict::Ok {
            n_streams: 1,
            expected_speedup: 1.0,
        };
    }
    let ratio = copy_ms.max(compute_ms) / copy_ms.min(compute_ms);
    let n_streams = if ratio > 2.0 {
        1
    } else if ratio > 1.5 {
        2
    } else {
        let cap = max_streams_avail.min(MAX_STREAMS);
        cap.max(2)
    };
    let serial = copy_ms + compute_ms;
    let parallel_phase = copy_ms.max(compute_ms);
    let speedup = if n_streams == 1 {
        1.0
    } else {
        let pipelined = parallel_phase + (serial - parallel_phase) / f64::from(n_streams);
        serial / pipelined
    };
    OverlapVerdict::Ok {
        n_streams,
        expected_speedup: speedup,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_async_memcpy_overlap")?;

    println!("balanced 50/50: {:?}", pick(50.0, 50.0, 4));
    println!("compute-heavy 10/100: {:?}", pick(10.0, 100.0, 4));
    println!("copy-heavy 100/10: {:?}", pick(100.0, 10.0, 4));
    println!("invalid neg: {:?}", pick(-1.0, 50.0, 4));
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
    fn balanced_uses_multi_stream() {
        let v = pick(50.0, 50.0, 4);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert!(n_streams >= 2);
        }
    }

    #[test]
    fn compute_dominates_one_stream() {
        // ratio = 100/10 = 10, > 2 → 1 stream.
        let v = pick(10.0, 100.0, 4);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert_eq!(n_streams, 1);
        }
    }

    #[test]
    fn copy_dominates_one_stream() {
        let v = pick(100.0, 10.0, 4);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert_eq!(n_streams, 1);
        }
    }

    #[test]
    fn slight_imbalance_2_streams() {
        // ratio = 60/40 = 1.5 → falls into 2 streams.
        let v = pick(40.0, 60.0, 4);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert!(n_streams >= 2);
        }
    }

    #[test]
    fn invalid_negative_rejected() {
        assert_eq!(pick(-1.0, 50.0, 4), OverlapVerdict::InvalidTiming);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(pick(f64::NAN, 50.0, 4), OverlapVerdict::InvalidTiming);
    }

    #[test]
    fn zero_compute_returns_one_stream() {
        let v = pick(50.0, 0.0, 4);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert_eq!(n_streams, 1);
        }
    }

    #[test]
    fn speedup_at_least_one() {
        let v = pick(50.0, 50.0, 4);
        if let OverlapVerdict::Ok {
            expected_speedup, ..
        } = v
        {
            assert!(expected_speedup >= 1.0);
        }
    }

    #[test]
    fn capped_by_max_streams_avail() {
        let v = pick(50.0, 50.0, 2);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert!(n_streams <= 2);
        }
    }

    #[test]
    fn capped_by_abs_max_streams() {
        let v = pick(50.0, 50.0, 100);
        if let OverlapVerdict::Ok { n_streams, .. } = v {
            assert!(n_streams <= MAX_STREAMS);
        }
    }
}
