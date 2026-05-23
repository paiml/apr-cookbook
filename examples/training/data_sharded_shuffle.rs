//! # Recipe: Sharded-Shuffle Data Pipeline
//!
//! **Category**: training
//! **CLI Equivalent**: `apr data shuffle --shards 8 --seed 42 --in data/ --out data-shuffled/`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example data_sharded_shuffle` exits 0
//! 2. [x] `cargo test --example data_sharded_shuffle` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr data` sharding in-process (no shell-out)
//! 10. [x] Unit tests cover shard balance, no-duplicates, determinism
//!
//! ## Learning Objective
//! Implements a two-pass sharded shuffle in the style of MapReduce: distribute
//! records across shards by modulo-hash, then shuffle each shard with the
//! deterministic RNG. Verifies record-count conservation and shard-size balance.
//!
//! ## Run Command
//! ```bash
//! cargo run --example data_sharded_shuffle
//! ```
//!
//! ## References
//! - Dean, J. & Ghemawat, S. (2008). *MapReduce: Simplified Data Processing on Large Clusters*. CACM. DOI: 10.1145/1327452.1327492

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::seq::SliceRandom;
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    id: u64,
    payload: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ShardStats {
    index: usize,
    len: usize,
    first_id: Option<u64>,
    last_id: Option<u64>,
}

// ---------------------------------------------------------------------------
// Sharding logic
// ---------------------------------------------------------------------------

fn generate_records(seed: u64, n: usize) -> Vec<Record> {
    let bytes = generate_model_payload(seed, n * 16);
    (0..n)
        .map(|i| Record {
            id: i as u64,
            payload: bytes[i * 16..(i + 1) * 16].to_vec(),
        })
        .collect()
}

fn shard_by_hash(records: &[Record], n_shards: usize) -> Vec<Vec<Record>> {
    let mut shards: Vec<Vec<Record>> = (0..n_shards).map(|_| Vec::new()).collect();
    for r in records {
        let target = (r.id as usize) % n_shards.max(1);
        shards[target].push(r.clone());
    }
    shards
}

fn shuffle_each_shard(shards: &mut [Vec<Record>], rng: &mut rand::rngs::StdRng) {
    for s in shards.iter_mut() {
        s.shuffle(rng);
    }
}

fn shard_stats(shards: &[Vec<Record>]) -> Vec<ShardStats> {
    shards
        .iter()
        .enumerate()
        .map(|(i, s)| ShardStats {
            index: i,
            len: s.len(),
            first_id: s.first().map(|r| r.id),
            last_id: s.last().map(|r| r.id),
        })
        .collect()
}

/// Count how many records are in the output. Must equal the input count.
fn total_records(shards: &[Vec<Record>]) -> usize {
    shards.iter().map(Vec::len).sum()
}

/// Maximum size - minimum size across shards. Ideally small for balance.
fn shard_imbalance(shards: &[Vec<Record>]) -> usize {
    let sizes: Vec<usize> = shards.iter().map(Vec::len).collect();
    let max = sizes.iter().max().copied().unwrap_or(0);
    let min = sizes.iter().min().copied().unwrap_or(0);
    max.saturating_sub(min)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("data_sharded_shuffle")?;
    println!("=== Recipe: {} ===", ctx.name());

    let n_records = 200_usize;
    let n_shards = 8_usize;
    let records = generate_records(hash_name_to_seed("data-sharded-shuffle"), n_records);
    println!(
        "Generated {} records; sharding across {} shards",
        records.len(),
        n_shards
    );

    let mut shards = shard_by_hash(&records, n_shards);
    shuffle_each_shard(&mut shards, ctx.rng());

    let stats = shard_stats(&shards);
    let total = total_records(&shards);
    let imbalance = shard_imbalance(&shards);

    println!("\n--- Shard Stats ---");
    println!(
        "{:>6} {:>10} {:>12} {:>12}",
        "Shard", "Size", "FirstID", "LastID"
    );
    for s in &stats {
        println!(
            "{:>6} {:>10} {:>12} {:>12}",
            s.index,
            s.len,
            s.first_id.map_or(-1, |v| v as i64),
            s.last_id.map_or(-1, |v| v as i64),
        );
    }
    println!("\nTotal records: {total}");
    println!("Imbalance (max - min): {imbalance}");

    assert_eq!(total, records.len(), "sharding must preserve record count");
    assert!(
        imbalance <= n_records / n_shards,
        "shards should be roughly balanced"
    );

    let out = json!({
        "recipe": ctx.name(),
        "n_records": n_records,
        "n_shards": n_shards,
        "total_after_sharding": total,
        "imbalance": imbalance,
        "shard_sizes": stats.iter().map(|s| s.len).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("shard-stats.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.record_metric("n_records", n_records as i64);
    ctx.record_metric("n_shards", n_shards as i64);
    ctx.record_metric("imbalance", imbalance as i64);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_sharding_preserves_count() {
        let recs = generate_records(1, 100);
        let shards = shard_by_hash(&recs, 4);
        assert_eq!(total_records(&shards), 100);
    }

    #[test]
    fn test_sharding_no_duplicates() {
        let recs = generate_records(2, 50);
        let shards = shard_by_hash(&recs, 5);
        let mut all_ids: Vec<u64> = shards.iter().flatten().map(|r| r.id).collect();
        all_ids.sort_unstable();
        let original: Vec<u64> = (0..50).collect();
        assert_eq!(all_ids, original);
    }

    #[test]
    fn test_sharding_balance() {
        let recs = generate_records(3, 160);
        let shards = shard_by_hash(&recs, 8);
        // With mod-hash on sequential IDs, perfectly balanced.
        assert_eq!(shard_imbalance(&shards), 0);
    }

    #[test]
    fn test_shuffle_is_deterministic_for_same_seed() {
        let recs = generate_records(4, 64);
        let mut a = shard_by_hash(&recs, 4);
        let mut b = shard_by_hash(&recs, 4);
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(99);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(99);
        shuffle_each_shard(&mut a, &mut rng_a);
        shuffle_each_shard(&mut b, &mut rng_b);
        for (x, y) in a.iter().zip(b.iter()) {
            let xi: Vec<u64> = x.iter().map(|r| r.id).collect();
            let yi: Vec<u64> = y.iter().map(|r| r.id).collect();
            assert_eq!(xi, yi);
        }
    }

    #[test]
    fn test_single_shard_contains_everything() {
        let recs = generate_records(5, 30);
        let shards = shard_by_hash(&recs, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].len(), 30);
    }

    #[test]
    fn test_empty_input_empty_shards() {
        let shards = shard_by_hash(&[], 4);
        assert_eq!(shards.len(), 4);
        assert!(shards.iter().all(|s| s.is_empty()));
    }
}
