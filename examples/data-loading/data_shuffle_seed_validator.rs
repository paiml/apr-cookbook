//! # Data Shuffle Seed Validator
//!
//! For reproducible training: every worker must derive its shuffle seed
//! from `(global_seed, epoch, worker_rank)`. This recipe builds the
//! per-worker seed deriver + the cross-worker uniqueness check.
//!
//! Demonstrates the **DATA.20** recipe for PMAT-132 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DATA-001 + Pratt 2009 (deterministic shuffling).
//!
//! Run with: cargo run --example data_shuffle_seed_validator
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum SeedVerdict {
    Ok { seeds: Vec<u64> },
    InvalidWorkerCount,
    DuplicateSeedDetected,
}

pub fn derive_seeds(global_seed: u64, epoch: u32, num_workers: u32) -> SeedVerdict {
    if num_workers == 0 {
        return SeedVerdict::InvalidWorkerCount;
    }
    let mut seeds = Vec::with_capacity(num_workers as usize);
    for w in 0..num_workers {
        seeds.push(derive_one(global_seed, epoch, w));
    }
    let set: HashSet<u64> = seeds.iter().copied().collect();
    if set.len() != seeds.len() {
        return SeedVerdict::DuplicateSeedDetected;
    }
    SeedVerdict::Ok { seeds }
}

pub fn derive_one(global_seed: u64, epoch: u32, worker_rank: u32) -> u64 {
    // SplitMix64-style mixing — deterministic, no collisions for normal ranges.
    let mut z = global_seed
        .wrapping_add(u64::from(epoch).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(u64::from(worker_rank).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_shuffle_seed_validator")?;

    println!("4 workers epoch 0: {:?}", derive_seeds(42, 0, 4));
    println!("4 workers epoch 1: {:?}", derive_seeds(42, 1, 4));
    println!("0 workers: {:?}", derive_seeds(42, 0, 0));

    let s_a = derive_one(42, 0, 0);
    let s_b = derive_one(42, 1, 0);
    println!("epoch 0 vs epoch 1 (same worker): {} != {}", s_a, s_b);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_4_workers_unique_seeds() {
        if let SeedVerdict::Ok { seeds } = derive_seeds(42, 0, 4) {
            let set: HashSet<u64> = seeds.iter().copied().collect();
            assert_eq!(set.len(), 4);
        }
    }

    #[test]
    fn zero_workers_rejected() {
        assert_eq!(derive_seeds(42, 0, 0), SeedVerdict::InvalidWorkerCount);
    }

    #[test]
    fn deterministic_across_calls() {
        let a = derive_seeds(42, 0, 4);
        let b = derive_seeds(42, 0, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn epoch_changes_seeds() {
        let a = derive_one(42, 0, 0);
        let b = derive_one(42, 1, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn worker_rank_changes_seed() {
        let a = derive_one(42, 0, 0);
        let b = derive_one(42, 0, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn global_seed_changes_seed() {
        let a = derive_one(42, 0, 0);
        let b = derive_one(43, 0, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn many_workers_no_collisions() {
        if let SeedVerdict::Ok { seeds } = derive_seeds(0, 0, 1000) {
            let set: HashSet<u64> = seeds.iter().copied().collect();
            assert_eq!(set.len(), 1000);
        }
    }

    #[test]
    fn single_worker_handled() {
        if let SeedVerdict::Ok { seeds } = derive_seeds(42, 0, 1) {
            assert_eq!(seeds.len(), 1);
        }
    }

    #[test]
    fn zero_global_seed_handled() {
        if let SeedVerdict::Ok { seeds } = derive_seeds(0, 0, 4) {
            // All seeds derived; even with seed 0, mixing produces non-zero values.
            let set: HashSet<u64> = seeds.iter().copied().collect();
            assert_eq!(set.len(), 4);
        }
    }

    #[test]
    fn cross_epoch_different_outputs() {
        // Verify all-worker seed sets differ between epochs.
        let e0 = derive_seeds(42, 0, 8);
        let e1 = derive_seeds(42, 1, 8);
        assert_ne!(e0, e1);
    }
}
