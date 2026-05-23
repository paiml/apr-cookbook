//! # Inference KV-Cache LRU Eviction
//!
//! When KV-cache is full and a new prefix arrives, evict the
//! least-recently-used cached prefix. Tracks (prefix_hash, last_used).
//! This recipe builds the cache structure + insertion + eviction
//! decision.
//!
//! Demonstrates the **INF.12** recipe for PMAT-129 (inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Belady (1966). A study of replacement algorithms for a virtual-storage computer.
//!
//! Run with: cargo run --example inference_kv_cache_lru
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Insert(u64, u64), // (prefix_hash, used_at_ms)
    Lookup(u64, u64), // (prefix_hash, lookup_at_ms)
}

#[derive(Debug, PartialEq)]
pub enum CacheEvent {
    Inserted,
    HitRefreshed,
    Evicted { evicted_hash: u64 },
    Miss,
    InvalidCapacity,
}

pub struct LruCache {
    capacity: usize,
    entries: Vec<(u64, u64)>,
}

impl LruCache {
    pub fn new(capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return None;
        }
        Some(Self {
            capacity,
            entries: Vec::with_capacity(capacity),
        })
    }

    pub fn apply(&mut self, op: Op) -> CacheEvent {
        match op {
            Op::Lookup(hash, now) => {
                if let Some(idx) = self.entries.iter().position(|(h, _)| *h == hash) {
                    self.entries[idx].1 = now;
                    CacheEvent::HitRefreshed
                } else {
                    CacheEvent::Miss
                }
            }
            Op::Insert(hash, now) => {
                if let Some(idx) = self.entries.iter().position(|(h, _)| *h == hash) {
                    self.entries[idx].1 = now;
                    return CacheEvent::HitRefreshed;
                }
                if self.entries.len() >= self.capacity {
                    let (lru_idx, _) = self
                        .entries
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, t))| *t)
                        .unwrap();
                    let evicted_hash = self.entries[lru_idx].0;
                    self.entries[lru_idx] = (hash, now);
                    return CacheEvent::Evicted { evicted_hash };
                }
                self.entries.push((hash, now));
                CacheEvent::Inserted
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_kv_cache_lru")?;

    let mut cache = LruCache::new(2).unwrap();
    println!("ins A: {:?}", cache.apply(Op::Insert(0xa, 100)));
    println!("ins B: {:?}", cache.apply(Op::Insert(0xb, 200)));
    println!("look A: {:?}", cache.apply(Op::Lookup(0xa, 300)));
    println!("ins C: {:?}", cache.apply(Op::Insert(0xc, 400)));
    println!("len: {}", cache.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_capacity_invalid() {
        assert!(LruCache::new(0).is_none());
    }

    #[test]
    fn insert_into_empty_cache() {
        let mut c = LruCache::new(2).unwrap();
        assert_eq!(c.apply(Op::Insert(1, 100)), CacheEvent::Inserted);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn duplicate_insert_refreshes_timestamp() {
        let mut c = LruCache::new(2).unwrap();
        c.apply(Op::Insert(1, 100));
        let v = c.apply(Op::Insert(1, 200));
        assert_eq!(v, CacheEvent::HitRefreshed);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn lookup_hit_refreshes_timestamp() {
        let mut c = LruCache::new(2).unwrap();
        c.apply(Op::Insert(1, 100));
        let v = c.apply(Op::Lookup(1, 200));
        assert_eq!(v, CacheEvent::HitRefreshed);
    }

    #[test]
    fn lookup_miss_returns_miss() {
        let mut c = LruCache::new(2).unwrap();
        assert_eq!(c.apply(Op::Lookup(99, 100)), CacheEvent::Miss);
    }

    #[test]
    fn eviction_when_full() {
        let mut c = LruCache::new(2).unwrap();
        c.apply(Op::Insert(1, 100));
        c.apply(Op::Insert(2, 200));
        // Cache full; insert 3 evicts entry 1 (LRU).
        let v = c.apply(Op::Insert(3, 300));
        assert!(matches!(v, CacheEvent::Evicted { evicted_hash: 1 }));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn refresh_protects_from_eviction() {
        let mut c = LruCache::new(2).unwrap();
        c.apply(Op::Insert(1, 100));
        c.apply(Op::Insert(2, 200));
        // Refresh entry 1.
        c.apply(Op::Lookup(1, 250));
        // Now insert 3; entry 2 is now LRU.
        let v = c.apply(Op::Insert(3, 300));
        assert!(matches!(v, CacheEvent::Evicted { evicted_hash: 2 }));
    }

    #[test]
    fn capacity_one_evicts_immediately() {
        let mut c = LruCache::new(1).unwrap();
        c.apply(Op::Insert(1, 100));
        let v = c.apply(Op::Insert(2, 200));
        assert!(matches!(v, CacheEvent::Evicted { evicted_hash: 1 }));
    }

    #[test]
    fn is_empty_initially_true() {
        let c = LruCache::new(2).unwrap();
        assert!(c.is_empty());
    }
}
