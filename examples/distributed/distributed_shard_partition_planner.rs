//! # Distributed Shard Partition Planner
//!
//! Data-parallel training/inference splits a corpus into N shards, one
//! per worker. Constraints: shards must be balanced (size delta ≤ 1
//! sample) and contiguous within each worker. This recipe builds the
//! planner: input N samples × W workers → per-worker (start, len) ranges.
//!
//! Demonstrates the **DIST.2** recipe for PMAT-124 (distributed coverage —
//! closing F-invariant gap from 1 → 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Dean & Ghemawat (2004). MapReduce: Simplified Data Processing on Large Clusters.
//!
//! Run with: cargo run --example distributed_shard_partition_planner
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Shard {
    pub start: usize,
    pub len: usize,
}

#[derive(Debug, PartialEq)]
pub enum PartitionVerdict {
    Ok(Vec<Shard>),
    NoSamples,
    NoWorkers,
    MoreWorkersThanSamples { workers: usize, samples: usize },
}

pub fn plan(num_samples: usize, num_workers: usize) -> PartitionVerdict {
    if num_samples == 0 {
        return PartitionVerdict::NoSamples;
    }
    if num_workers == 0 {
        return PartitionVerdict::NoWorkers;
    }
    if num_workers > num_samples {
        return PartitionVerdict::MoreWorkersThanSamples {
            workers: num_workers,
            samples: num_samples,
        };
    }
    let base = num_samples / num_workers;
    let remainder = num_samples % num_workers;
    let mut shards = Vec::with_capacity(num_workers);
    let mut offset = 0;
    for w in 0..num_workers {
        let len = if w < remainder { base + 1 } else { base };
        shards.push(Shard { start: offset, len });
        offset += len;
    }
    PartitionVerdict::Ok(shards)
}

pub fn balance_delta(shards: &[Shard]) -> usize {
    let max = shards.iter().map(|s| s.len).max().unwrap_or(0);
    let min = shards.iter().map(|s| s.len).min().unwrap_or(0);
    max - min
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_shard_partition_planner")?;

    for (n, w) in [(100usize, 4usize), (10, 3), (5, 5), (3, 5), (0, 4), (10, 0)] {
        println!("samples={n} workers={w}  →  {:?}", plan(n, w));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_division_zero_remainder() {
        // 100 / 4 = 25 exactly.
        if let PartitionVerdict::Ok(shards) = plan(100, 4) {
            assert_eq!(shards.len(), 4);
            assert!(shards.iter().all(|s| s.len == 25));
            assert_eq!(balance_delta(&shards), 0);
        }
    }

    #[test]
    fn uneven_division_fronts_get_extras() {
        // 10 / 3 = 3 r 1 → first shard gets 4, others get 3.
        if let PartitionVerdict::Ok(shards) = plan(10, 3) {
            assert_eq!(shards.len(), 3);
            assert_eq!(shards[0].len, 4);
            assert_eq!(shards[1].len, 3);
            assert_eq!(shards[2].len, 3);
            assert_eq!(balance_delta(&shards), 1);
        }
    }

    #[test]
    fn shards_are_contiguous_no_gaps() {
        if let PartitionVerdict::Ok(shards) = plan(100, 7) {
            // start[i+1] = start[i] + len[i].
            for w in shards.windows(2) {
                assert_eq!(w[0].start + w[0].len, w[1].start);
            }
        }
    }

    #[test]
    fn shards_cover_all_samples() {
        if let PartitionVerdict::Ok(shards) = plan(100, 7) {
            let total: usize = shards.iter().map(|s| s.len).sum();
            assert_eq!(total, 100);
        }
    }

    #[test]
    fn single_worker_owns_everything() {
        if let PartitionVerdict::Ok(shards) = plan(50, 1) {
            assert_eq!(shards.len(), 1);
            assert_eq!(shards[0].start, 0);
            assert_eq!(shards[0].len, 50);
        }
    }

    #[test]
    fn equal_workers_and_samples_one_each() {
        if let PartitionVerdict::Ok(shards) = plan(5, 5) {
            assert!(shards.iter().all(|s| s.len == 1));
        }
    }

    #[test]
    fn more_workers_than_samples_rejected() {
        let v = plan(3, 5);
        assert!(matches!(v, PartitionVerdict::MoreWorkersThanSamples { .. }));
    }

    #[test]
    fn zero_samples_rejected() {
        assert_eq!(plan(0, 4), PartitionVerdict::NoSamples);
    }

    #[test]
    fn zero_workers_rejected() {
        assert_eq!(plan(10, 0), PartitionVerdict::NoWorkers);
    }

    #[test]
    fn balance_delta_at_most_one() {
        // Property: any valid plan has max-min ≤ 1.
        for n in [10usize, 17, 23, 100, 1000] {
            for w in [1usize, 3, 7, 16] {
                if w > n {
                    continue;
                }
                if let PartitionVerdict::Ok(shards) = plan(n, w) {
                    assert!(balance_delta(&shards) <= 1, "n={n} w={w}");
                }
            }
        }
    }
}
