//! # Distributed Ring-AllReduce Chunk Sizer
//!
//! Ring all-reduce splits the message into N chunks (N = workers).
//! Each step transfers chunk_size bytes over the slowest link.
//! Optimal chunk_size = message_size / num_workers.
//!
//! Plus tail-handling: if message_size % N != 0, last chunk is smaller.
//!
//! Demonstrates the **DIST.16** recipe for PMAT-150 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Patarasuk & Yuan (2009). Bandwidth Optimal All-Reduce on Tree Topologies.
//!
//! Run with: cargo run --example distributed_ring_chunks
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChunkVerdict {
    Ok {
        chunk_bytes: u64,
        last_chunk_bytes: u64,
        rounds: u32,
    },
    InvalidWorkers,
    InvalidMessage,
}

pub fn pick(message_bytes: u64, num_workers: u32) -> ChunkVerdict {
    if num_workers < 2 {
        return ChunkVerdict::InvalidWorkers;
    }
    if message_bytes == 0 {
        return ChunkVerdict::InvalidMessage;
    }
    let n = u64::from(num_workers);
    let chunk_bytes = message_bytes / n;
    let remainder = message_bytes % n;
    let last_chunk_bytes = if remainder == 0 {
        chunk_bytes
    } else {
        remainder
    };
    let rounds = 2 * (num_workers - 1);
    ChunkVerdict::Ok {
        chunk_bytes,
        last_chunk_bytes,
        rounds,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_ring_chunks")?;

    println!("100 MiB / 8 workers: {:?}", pick(100 * 1024 * 1024, 8));
    println!("100 MiB / 7 workers: {:?}", pick(100 * 1024 * 1024, 7));
    println!("100 bytes / 4 workers: {:?}", pick(100, 4));
    println!("invalid 1 worker: {:?}", pick(1024, 1));
    println!("invalid 0 message: {:?}", pick(0, 8));
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
    fn even_split() {
        // 100 MiB / 8 = 12.5 MiB chunks.
        let v = pick(100 * 1024 * 1024, 8);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 100 * 1024 * 1024 / 8);
        }
    }

    #[test]
    fn uneven_split_handles_remainder() {
        // 100 / 7 = 14 chunks + remainder 2.
        let v = pick(100, 7);
        if let ChunkVerdict::Ok {
            chunk_bytes,
            last_chunk_bytes,
            ..
        } = v
        {
            assert_eq!(chunk_bytes, 14);
            assert_eq!(last_chunk_bytes, 2);
        }
    }

    #[test]
    fn invalid_one_worker() {
        assert_eq!(pick(1024, 1), ChunkVerdict::InvalidWorkers);
    }

    #[test]
    fn invalid_zero_workers() {
        assert_eq!(pick(1024, 0), ChunkVerdict::InvalidWorkers);
    }

    #[test]
    fn invalid_zero_message() {
        assert_eq!(pick(0, 8), ChunkVerdict::InvalidMessage);
    }

    #[test]
    fn rounds_proportional_to_workers() {
        let v_4 = pick(1024, 4);
        let v_8 = pick(1024, 8);
        if let (ChunkVerdict::Ok { rounds: r4, .. }, ChunkVerdict::Ok { rounds: r8, .. }) =
            (v_4, v_8)
        {
            assert!(r8 > r4);
        }
    }

    #[test]
    fn small_message_one_byte_per_chunk() {
        // 8 bytes / 8 workers = 1 byte per chunk.
        let v = pick(8, 8);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 1);
        }
    }

    #[test]
    fn rounds_formula_correct() {
        // 2 × (N - 1).
        let v = pick(1024, 8);
        if let ChunkVerdict::Ok { rounds, .. } = v {
            assert_eq!(rounds, 14);
        }
    }

    #[test]
    fn last_equals_first_when_evenly_divisible() {
        let v = pick(800, 8);
        if let ChunkVerdict::Ok {
            chunk_bytes,
            last_chunk_bytes,
            ..
        } = v
        {
            assert_eq!(chunk_bytes, last_chunk_bytes);
        }
    }

    #[test]
    fn min_workers_two_succeeds() {
        let v = pick(100, 2);
        assert!(matches!(v, ChunkVerdict::Ok { .. }));
    }

    #[test]
    fn message_smaller_than_workers() {
        // 5 bytes / 8 workers → 0 byte chunks + remainder.
        let v = pick(5, 8);
        if let ChunkVerdict::Ok {
            chunk_bytes,
            last_chunk_bytes,
            ..
        } = v
        {
            assert_eq!(chunk_bytes, 0);
            assert_eq!(last_chunk_bytes, 5);
        }
    }
}
