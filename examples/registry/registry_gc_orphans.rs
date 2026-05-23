//! # Registry Garbage-Collect Orphan Tensors
//!
//! Content-addressable tensors are shared across bundles. After bundle
//! deletion, some tensors lose all references → orphans. GC: union all
//! referenced tensor hashes, then take the set difference of stored ↔
//! referenced.
//!
//! This recipe builds the GC planner: returns orphan hashes + bytes
//! freed.
//!
//! Demonstrates the **REG.14** recipe for PMAT-138 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: git gc — pruning unreachable objects.
//!
//! Run with: cargo run --example registry_gc_orphans
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum GcVerdict {
    Ok {
        orphan_hashes: Vec<String>,
        bytes_freed: u64,
        kept_count: usize,
    },
    EmptyStore,
}

pub fn plan(stored: &BTreeMap<String, u64>, bundle_refs: &[Vec<String>]) -> GcVerdict {
    if stored.is_empty() {
        return GcVerdict::EmptyStore;
    }
    let mut referenced: BTreeSet<&str> = BTreeSet::new();
    for bundle in bundle_refs {
        for h in bundle {
            referenced.insert(h.as_str());
        }
    }
    let mut orphan_hashes = Vec::new();
    let mut bytes_freed = 0u64;
    let mut kept_count = 0usize;
    for (hash, size) in stored {
        if referenced.contains(hash.as_str()) {
            kept_count += 1;
        } else {
            orphan_hashes.push(hash.clone());
            bytes_freed += *size;
        }
    }
    GcVerdict::Ok {
        orphan_hashes,
        bytes_freed,
        kept_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_gc_orphans")?;

    let mut stored = BTreeMap::new();
    stored.insert("hash_a".to_string(), 1000);
    stored.insert("hash_b".to_string(), 2000);
    stored.insert("hash_c".to_string(), 500);
    stored.insert("hash_orphan".to_string(), 800);

    let refs = vec![
        vec!["hash_a".to_string(), "hash_b".to_string()],
        vec!["hash_c".to_string()],
    ];
    println!("typical: {:?}", plan(&stored, &refs));
    println!("empty store: {:?}", plan(&BTreeMap::new(), &refs));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }

    fn refs(bundles: &[&[&str]]) -> Vec<Vec<String>> {
        bundles
            .iter()
            .map(|b| b.iter().map(|s| (*s).to_string()).collect())
            .collect()
    }

    #[test]
    fn gc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_orphans_when_all_referenced() {
        let s = store(&[("a", 100), ("b", 200)]);
        let r = refs(&[&["a", "b"]]);
        if let GcVerdict::Ok {
            orphan_hashes,
            bytes_freed,
            ..
        } = plan(&s, &r)
        {
            assert!(orphan_hashes.is_empty());
            assert_eq!(bytes_freed, 0);
        }
    }

    #[test]
    fn unreferenced_become_orphans() {
        let s = store(&[("a", 100), ("orphan", 999)]);
        let r = refs(&[&["a"]]);
        if let GcVerdict::Ok {
            orphan_hashes,
            bytes_freed,
            ..
        } = plan(&s, &r)
        {
            assert_eq!(orphan_hashes, vec!["orphan".to_string()]);
            assert_eq!(bytes_freed, 999);
        }
    }

    #[test]
    fn empty_store_rejected() {
        let r = refs(&[&["a"]]);
        assert_eq!(plan(&BTreeMap::new(), &r), GcVerdict::EmptyStore);
    }

    #[test]
    fn no_bundles_all_orphan() {
        let s = store(&[("a", 100), ("b", 200)]);
        let r = refs(&[]);
        if let GcVerdict::Ok {
            orphan_hashes,
            bytes_freed,
            ..
        } = plan(&s, &r)
        {
            assert_eq!(orphan_hashes.len(), 2);
            assert_eq!(bytes_freed, 300);
        }
    }

    #[test]
    fn shared_tensor_kept_once() {
        // Same hash referenced by 3 bundles → still 1 kept.
        let s = store(&[("shared", 500)]);
        let r = refs(&[&["shared"], &["shared"], &["shared"]]);
        if let GcVerdict::Ok { kept_count, .. } = plan(&s, &r) {
            assert_eq!(kept_count, 1);
        }
    }

    #[test]
    fn kept_count_correct() {
        let s = store(&[("a", 100), ("b", 200), ("c", 300)]);
        let r = refs(&[&["a"], &["b"]]);
        if let GcVerdict::Ok { kept_count, .. } = plan(&s, &r) {
            assert_eq!(kept_count, 2);
        }
    }

    #[test]
    fn dangling_ref_in_bundle_not_an_error() {
        // Bundle references hash not in store → just no-op.
        let s = store(&[("a", 100)]);
        let r = refs(&[&["a", "ghost"]]);
        if let GcVerdict::Ok { orphan_hashes, .. } = plan(&s, &r) {
            assert!(orphan_hashes.is_empty());
        }
    }

    #[test]
    fn bytes_freed_sums_across_orphans() {
        let s = store(&[("a", 100), ("b", 200), ("c", 300)]);
        let r = refs(&[]);
        if let GcVerdict::Ok { bytes_freed, .. } = plan(&s, &r) {
            assert_eq!(bytes_freed, 600);
        }
    }

    #[test]
    fn multiple_bundles_union_correctly() {
        let s = store(&[("a", 100), ("b", 200), ("c", 300), ("d", 400)]);
        let r = refs(&[&["a", "b"], &["c"]]);
        if let GcVerdict::Ok { orphan_hashes, .. } = plan(&s, &r) {
            assert_eq!(orphan_hashes, vec!["d".to_string()]);
        }
    }

    #[test]
    fn orphan_hashes_sorted() {
        // BTreeMap iteration is sorted → orphans collected in lex order.
        let s = store(&[("zzz", 100), ("aaa", 200), ("mmm", 300)]);
        let r = refs(&[]);
        if let GcVerdict::Ok { orphan_hashes, .. } = plan(&s, &r) {
            assert_eq!(orphan_hashes, vec!["aaa", "mmm", "zzz"]);
        }
    }
}
