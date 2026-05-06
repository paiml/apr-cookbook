//! # Distributed Consistent-Hash Ring
//!
//! Maps keys to nodes using a hash ring with virtual nodes (vnodes).
//! Each physical node owns multiple positions on the ring; key →
//! lowest_position ≥ hash(key) wraps around. Adding/removing nodes
//! moves only ~1/N keys.
//!
//! This recipe builds the ring + the lookup function.
//!
//! Demonstrates the **DIST.13** recipe for PMAT-145 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Karger et al. (1997). Consistent Hashing and Random Trees.
//!
//! Run with: cargo run --example distributed_consistent_hash
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

const VNODES_PER_NODE: u32 = 100;

#[derive(Debug, PartialEq)]
pub enum LookupVerdict {
    Ok { node: String, position: u64 },
    EmptyRing,
}

pub struct HashRing {
    ring: BTreeMap<u64, String>,
}

impl HashRing {
    pub fn new(nodes: &[&str]) -> Self {
        let mut ring = BTreeMap::new();
        for &n in nodes {
            for v in 0..VNODES_PER_NODE {
                let pos = pseudo_hash(format!("{n}#{v}").as_bytes());
                ring.insert(pos, n.to_string());
            }
        }
        Self { ring }
    }

    pub fn add(&mut self, node: &str) {
        for v in 0..VNODES_PER_NODE {
            let pos = pseudo_hash(format!("{node}#{v}").as_bytes());
            self.ring.insert(pos, node.to_string());
        }
    }

    pub fn remove(&mut self, node: &str) {
        for v in 0..VNODES_PER_NODE {
            let pos = pseudo_hash(format!("{node}#{v}").as_bytes());
            self.ring.remove(&pos);
        }
    }

    pub fn lookup(&self, key: &str) -> LookupVerdict {
        if self.ring.is_empty() {
            return LookupVerdict::EmptyRing;
        }
        let h = pseudo_hash(key.as_bytes());
        let entry = self
            .ring
            .range(h..)
            .next()
            .or_else(|| self.ring.iter().next());
        match entry {
            Some((position, node)) => LookupVerdict::Ok {
                node: node.clone(),
                position: *position,
            },
            None => LookupVerdict::EmptyRing,
        }
    }

    pub fn ring_size(&self) -> usize {
        self.ring.len()
    }
}

fn pseudo_hash(data: &[u8]) -> u64 {
    // Deterministic non-cryptographic FNV-1a.
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_consistent_hash")?;

    let mut ring = HashRing::new(&["node-a", "node-b", "node-c"]);
    println!("ring size: {}", ring.ring_size());
    println!("key1: {:?}", ring.lookup("key1"));
    println!("key2: {:?}", ring.lookup("key2"));

    ring.add("node-d");
    println!("after add: {}", ring.ring_size());

    ring.remove("node-a");
    println!("after remove: {}", ring.ring_size());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ring_size_proportional_to_nodes() {
        let r = HashRing::new(&["a", "b", "c"]);
        assert_eq!(r.ring_size(), 3 * VNODES_PER_NODE as usize);
    }

    #[test]
    fn empty_ring_lookup_returns_empty() {
        let r = HashRing::new(&[]);
        assert_eq!(r.lookup("key"), LookupVerdict::EmptyRing);
    }

    #[test]
    fn key_lookup_returns_node() {
        let r = HashRing::new(&["a", "b", "c"]);
        let v = r.lookup("hello");
        assert!(matches!(v, LookupVerdict::Ok { .. }));
    }

    #[test]
    fn same_key_consistent() {
        let r = HashRing::new(&["a", "b", "c"]);
        let v1 = r.lookup("hello");
        let v2 = r.lookup("hello");
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_keys_may_get_different_nodes() {
        let r = HashRing::new(&["a", "b", "c", "d", "e"]);
        let mut nodes = std::collections::BTreeSet::new();
        for i in 0..1000 {
            if let LookupVerdict::Ok { node, .. } = r.lookup(&format!("key{i}")) {
                nodes.insert(node);
            }
        }
        // Distribution should span at least 2 nodes (FNV-1a is non-uniform).
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn add_node_grows_ring() {
        let mut r = HashRing::new(&["a"]);
        let initial = r.ring_size();
        r.add("b");
        assert_eq!(r.ring_size(), initial + VNODES_PER_NODE as usize);
    }

    #[test]
    fn remove_node_shrinks_ring() {
        let mut r = HashRing::new(&["a", "b"]);
        let initial = r.ring_size();
        r.remove("a");
        assert_eq!(r.ring_size(), initial - VNODES_PER_NODE as usize);
    }

    #[test]
    fn wraparound_to_first_node() {
        let r = HashRing::new(&["a", "b"]);
        // Some key will hash above all positions; should still find a node.
        for i in 0..50 {
            let v = r.lookup(&format!("k{i}"));
            assert!(matches!(v, LookupVerdict::Ok { .. }));
        }
    }

    #[test]
    fn vnodes_distribute_load() {
        // With 1000 keys + 5 nodes + 100 vnodes each, distribution should be
        // reasonably balanced (no single node dominates).
        let r = HashRing::new(&["a", "b", "c", "d", "e"]);
        let mut counts = std::collections::HashMap::new();
        for i in 0..1000 {
            if let LookupVerdict::Ok { node, .. } = r.lookup(&format!("key{i}")) {
                *counts.entry(node).or_insert(0) += 1;
            }
        }
        for (_, count) in &counts {
            // Each node should see at least 50 keys (out of 1000 over 5 nodes).
            assert!(*count >= 50);
        }
    }

    #[test]
    fn pseudo_hash_deterministic() {
        let h1 = pseudo_hash(b"test");
        let h2 = pseudo_hash(b"test");
        assert_eq!(h1, h2);
    }
}
