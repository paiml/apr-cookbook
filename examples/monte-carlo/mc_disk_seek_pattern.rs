//! # Monte-Carlo Disk Seek Pattern
//!
//! Sim seek time for sequential vs random access patterns. Sequential
//! pays 1ms seek penalty per N reads; random pays it every read.
//! Returns total time and observed mean per-read latency.
//!
//! Demonstrates the **MC.44** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HDD seek-time analysis (Jim Gray's "5-minute rule").
//!
//! Run with: cargo run --example mc_disk_seek_pattern
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Random,
}

#[derive(Debug, PartialEq)]
pub enum SeekVerdict {
    Ok { total_ms: f64, mean_latency_ms: f64 },
    InvalidConfig,
}

pub fn simulate(
    pattern: AccessPattern,
    reads: u32,
    seek_ms: f64,
    transfer_ms: f64,
    block_size: u32,
) -> SeekVerdict {
    if !seek_ms.is_finite()
        || seek_ms < 0.0
        || !transfer_ms.is_finite()
        || transfer_ms < 0.0
        || reads == 0
        || block_size == 0
    {
        return SeekVerdict::InvalidConfig;
    }
    let total_ms = match pattern {
        AccessPattern::Sequential => {
            // One seek per block_size reads.
            let blocks = reads.div_ceil(block_size);
            f64::from(blocks) * seek_ms + f64::from(reads) * transfer_ms
        }
        AccessPattern::Random => {
            // Seek per read.
            f64::from(reads) * (seek_ms + transfer_ms)
        }
    };
    let mean_latency_ms = total_ms / f64::from(reads);
    SeekVerdict::Ok {
        total_ms,
        mean_latency_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_disk_seek_pattern")?;

    println!(
        "sequential: {:?}",
        simulate(AccessPattern::Sequential, 1000, 5.0, 0.1, 64)
    );
    println!(
        "random: {:?}",
        simulate(AccessPattern::Random, 1000, 5.0, 0.1, 64)
    );
    println!(
        "invalid: {:?}",
        simulate(AccessPattern::Random, 0, 5.0, 0.1, 64)
    );
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
    fn sequential_lower_than_random() {
        let s = simulate(AccessPattern::Sequential, 1000, 5.0, 0.1, 64);
        let r = simulate(AccessPattern::Random, 1000, 5.0, 0.1, 64);
        if let (SeekVerdict::Ok { total_ms: ts, .. }, SeekVerdict::Ok { total_ms: tr, .. }) = (s, r)
        {
            assert!(tr > ts);
        }
    }

    #[test]
    fn invalid_zero_reads() {
        assert_eq!(
            simulate(AccessPattern::Random, 0, 5.0, 0.1, 64),
            SeekVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_block() {
        assert_eq!(
            simulate(AccessPattern::Sequential, 100, 5.0, 0.1, 0),
            SeekVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_seek() {
        assert_eq!(
            simulate(AccessPattern::Random, 100, -1.0, 0.1, 64),
            SeekVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(AccessPattern::Random, 100, f64::NAN, 0.1, 64),
            SeekVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(AccessPattern::Sequential, 100, 5.0, 0.1, 64);
        let b = simulate(AccessPattern::Sequential, 100, 5.0, 0.1, 64);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_latency_positive() {
        let v = simulate(AccessPattern::Random, 100, 5.0, 0.1, 64);
        if let SeekVerdict::Ok {
            mean_latency_ms, ..
        } = v
        {
            assert!(mean_latency_ms > 0.0);
        }
    }

    #[test]
    fn random_amortizes_per_read() {
        let v = simulate(AccessPattern::Random, 1000, 5.0, 0.1, 64);
        if let SeekVerdict::Ok {
            mean_latency_ms, ..
        } = v
        {
            // Random: each read costs 5.1 ms.
            assert!((mean_latency_ms - 5.1).abs() < 1e-6);
        }
    }

    #[test]
    fn block_size_larger_than_reads() {
        // Sequential with block > reads → 1 seek total.
        let v = simulate(AccessPattern::Sequential, 5, 10.0, 1.0, 100);
        if let SeekVerdict::Ok { total_ms, .. } = v {
            // 1 seek (10) + 5 reads × 1 ms = 15.
            assert!((total_ms - 15.0).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_seek_pure_transfer() {
        let v = simulate(AccessPattern::Sequential, 100, 0.0, 1.0, 64);
        if let SeekVerdict::Ok { total_ms, .. } = v {
            assert!((total_ms - 100.0).abs() < 1e-9);
        }
    }
}
