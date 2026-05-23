//! # Monte-Carlo Random Set Partitions
//!
//! Sample N random partitions of a `n`-element set. Reports the
//! number of distinct partition shapes (block-size multiset) seen.
//!
//! Demonstrates the **MC.129** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bell number B(n) classic combinatorics; Stirling
//!  numbers of the 2nd kind.
//!
//! Run with: cargo run --example mc_random_partition_set
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum PartitionVerdict {
    Ok {
        distinct_shapes: u32,
        max_blocks: u32,
        min_blocks: u32,
    },
    InvalidConfig,
}

pub fn simulate(samples: u32, n: u32, seed: u64) -> PartitionVerdict {
    if samples == 0 || n == 0 || n > 20 {
        return PartitionVerdict::InvalidConfig;
    }
    let mut shapes: BTreeSet<Vec<u32>> = BTreeSet::new();
    let mut max_blocks: u32 = 0;
    let mut min_blocks: u32 = u32::MAX;
    let mut rng_state = seed | 1;
    for _ in 0..samples {
        // Random partition: assign each element to a random block index
        // chosen by a "coin flip" expansion (Crane-Towsley CRP-like).
        let mut assignment = vec![0u32; n as usize];
        let mut block_count = 1u32;
        for (elem, slot) in assignment.iter_mut().enumerate().skip(1) {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            // Pick existing block with prob proportional to size, else new.
            let new_block_prob = 1.0 / (1.0 + elem as f64);
            if r < new_block_prob {
                *slot = block_count;
                block_count += 1;
            } else {
                let pick = ((lcg(&mut rng_state) >> 32) as u32) % block_count;
                *slot = pick;
            }
        }
        let mut block_sizes: Vec<u32> = vec![0; block_count as usize];
        for &a in &assignment {
            block_sizes[a as usize] += 1;
        }
        block_sizes.sort_unstable();
        shapes.insert(block_sizes.clone());
        if block_count > max_blocks {
            max_blocks = block_count;
        }
        if block_count < min_blocks {
            min_blocks = block_count;
        }
    }
    PartitionVerdict::Ok {
        distinct_shapes: shapes.len() as u32,
        max_blocks,
        min_blocks,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_random_partition_set")?;

    println!("n=5: {:?}", simulate(1000, 5, 42));
    println!("n=10: {:?}", simulate(1000, 10, 42));
    println!("invalid: {:?}", simulate(0, 5, 42));
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
    fn invalid_zero_samples() {
        assert_eq!(simulate(0, 5, 42), PartitionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_n() {
        assert_eq!(simulate(100, 0, 42), PartitionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_n_above_20() {
        assert_eq!(simulate(100, 25, 42), PartitionVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 42);
        let b = simulate(500, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_shapes_le_bell_number() {
        // B(5) = 52.
        let v = simulate(2000, 5, 42);
        if let PartitionVerdict::Ok {
            distinct_shapes, ..
        } = v
        {
            assert!(distinct_shapes <= 52);
        }
    }

    #[test]
    fn min_blocks_at_least_one() {
        let v = simulate(500, 5, 42);
        if let PartitionVerdict::Ok { min_blocks, .. } = v {
            assert!(min_blocks >= 1);
        }
    }

    #[test]
    fn max_blocks_le_n() {
        let v = simulate(500, 5, 42);
        if let PartitionVerdict::Ok { max_blocks, .. } = v {
            assert!(max_blocks <= 5);
        }
    }

    #[test]
    fn max_ge_min() {
        let v = simulate(500, 5, 42);
        if let PartitionVerdict::Ok {
            max_blocks,
            min_blocks,
            ..
        } = v
        {
            assert!(max_blocks >= min_blocks);
        }
    }

    #[test]
    fn larger_n_more_shapes() {
        let small = simulate(2000, 3, 42);
        let big = simulate(2000, 7, 42);
        if let (
            PartitionVerdict::Ok {
                distinct_shapes: s, ..
            },
            PartitionVerdict::Ok {
                distinct_shapes: b, ..
            },
        ) = (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn n_one_only_one_shape() {
        let v = simulate(100, 1, 42);
        if let PartitionVerdict::Ok {
            distinct_shapes, ..
        } = v
        {
            assert_eq!(distinct_shapes, 1);
        }
    }

    #[test]
    fn many_samples_handled() {
        let v = simulate(5000, 8, 42);
        assert!(matches!(v, PartitionVerdict::Ok { .. }));
    }
}
