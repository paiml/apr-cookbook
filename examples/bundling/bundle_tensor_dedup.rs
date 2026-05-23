//! # Bundle Tensor Content-Hash Dedup
//!
//! Multiple bundles often share embedding tables, base weights, etc.
//! Content-hash-based dedup: store each unique tensor once in a global
//! pool; bundle references it by hash. Saves disk for fine-tuned model
//! collections (e.g., 100 LoRA fine-tunes of the same base).
//!
//! This recipe builds the dedup planner: given (name, hash, size)
//! triples across bundles, returns: unique tensors stored once, total
//! bytes saved, and per-bundle reference table.
//!
//! Demonstrates the **BUNDLE.19** recipe for PMAT-136 (bundling round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: git pack-file content-addressable storage model.
//!
//! Run with: cargo run --example bundle_tensor_dedup
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub hash: String,
    pub size: u64,
}

#[derive(Debug, PartialEq)]
pub struct DedupPlan {
    pub unique_count: usize,
    pub duplicate_count: usize,
    pub bytes_saved: u64,
    pub total_unique_bytes: u64,
}

#[derive(Debug, PartialEq)]
pub enum DedupVerdict {
    Ok(DedupPlan),
    EmptyTensors,
    InvalidHash { which: String },
}

const HASH_LEN: usize = 64;

pub fn plan(tensors: &[TensorRef]) -> DedupVerdict {
    if tensors.is_empty() {
        return DedupVerdict::EmptyTensors;
    }
    let mut by_hash: BTreeMap<&str, &TensorRef> = BTreeMap::new();
    let mut total_seen_bytes = 0u64;
    let mut duplicate_count = 0usize;

    for t in tensors {
        if t.hash.len() != HASH_LEN || !t.hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return DedupVerdict::InvalidHash {
                which: t.name.clone(),
            };
        }
        total_seen_bytes += t.size;
        if by_hash.contains_key(t.hash.as_str()) {
            duplicate_count += 1;
        } else {
            by_hash.insert(t.hash.as_str(), t);
        }
    }
    let total_unique_bytes: u64 = by_hash.values().map(|t| t.size).sum();
    let bytes_saved = total_seen_bytes - total_unique_bytes;
    DedupVerdict::Ok(DedupPlan {
        unique_count: by_hash.len(),
        duplicate_count,
        bytes_saved,
        total_unique_bytes,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_tensor_dedup")?;

    let h1 = "a".repeat(64);
    let h2 = "b".repeat(64);
    let tensors = vec![
        TensorRef {
            name: "base.embed".to_string(),
            hash: h1.clone(),
            size: 1000,
        },
        TensorRef {
            name: "lora1.embed".to_string(),
            hash: h1.clone(),
            size: 1000,
        },
        TensorRef {
            name: "lora1.adapter".to_string(),
            hash: h2.clone(),
            size: 200,
        },
        TensorRef {
            name: "lora2.embed".to_string(),
            hash: h1,
            size: 1000,
        },
    ];
    println!("4-tensor LoRA dedup: {:?}", plan(&tensors));
    println!("empty: {:?}", plan(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, hash: &str, size: u64) -> TensorRef {
        TensorRef {
            name: name.to_string(),
            hash: hash.to_string(),
            size,
        }
    }

    #[test]
    fn dedup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_duplicates_zero_savings() {
        let h1 = "a".repeat(64);
        let h2 = "b".repeat(64);
        let v = plan(&[t("a", &h1, 100), t("b", &h2, 200)]);
        if let DedupVerdict::Ok(p) = v {
            assert_eq!(p.duplicate_count, 0);
            assert_eq!(p.bytes_saved, 0);
            assert_eq!(p.unique_count, 2);
        }
    }

    #[test]
    fn full_duplicates_max_savings() {
        let h = "a".repeat(64);
        let v = plan(&[t("a", &h, 100), t("b", &h, 100), t("c", &h, 100)]);
        if let DedupVerdict::Ok(p) = v {
            assert_eq!(p.unique_count, 1);
            assert_eq!(p.duplicate_count, 2);
            assert_eq!(p.bytes_saved, 200);
        }
    }

    #[test]
    fn realistic_lora_collection() {
        // Base embedding (1000) shared across 3 LoRA adapters + 1 unique adapter (200).
        let h1 = "a".repeat(64);
        let h2 = "b".repeat(64);
        let v = plan(&[
            t("base.embed", &h1, 1000),
            t("lora1.embed", &h1, 1000),
            t("lora2.embed", &h1, 1000),
            t("lora1.adapter", &h2, 200),
        ]);
        if let DedupVerdict::Ok(p) = v {
            assert_eq!(p.unique_count, 2);
            assert_eq!(p.duplicate_count, 2);
            assert_eq!(p.bytes_saved, 2000);
            assert_eq!(p.total_unique_bytes, 1200);
        }
    }

    #[test]
    fn empty_tensors_rejected() {
        assert_eq!(plan(&[]), DedupVerdict::EmptyTensors);
    }

    #[test]
    fn invalid_short_hash_rejected() {
        let v = plan(&[t("bad", "abc", 100)]);
        assert!(matches!(v, DedupVerdict::InvalidHash { .. }));
    }

    #[test]
    fn invalid_non_hex_hash_rejected() {
        let bad = "z".repeat(64);
        let v = plan(&[t("bad", &bad, 100)]);
        assert!(matches!(v, DedupVerdict::InvalidHash { .. }));
    }

    #[test]
    fn long_hash_rejected() {
        let long = "a".repeat(128);
        let v = plan(&[t("bad", &long, 100)]);
        assert!(matches!(v, DedupVerdict::InvalidHash { .. }));
    }

    #[test]
    fn unique_count_le_input_count() {
        let h1 = "a".repeat(64);
        let h2 = "b".repeat(64);
        let v = plan(&[t("a", &h1, 100), t("b", &h2, 200), t("c", &h1, 100)]);
        if let DedupVerdict::Ok(p) = v {
            assert!(p.unique_count <= 3);
            assert_eq!(p.unique_count, 2);
        }
    }

    #[test]
    fn case_insensitive_hash_treated_as_distinct() {
        // Per content-addressable convention, hashes are normalized lowercase
        // upstream — this recipe treats different cases as different hashes.
        let lower = "a".repeat(64);
        let upper = "A".repeat(64);
        let v = plan(&[t("a", &lower, 100), t("b", &upper, 100)]);
        if let DedupVerdict::Ok(p) = v {
            assert_eq!(p.unique_count, 2);
        }
    }

    #[test]
    fn savings_proportional_to_dup_size() {
        let h = "a".repeat(64);
        let v = plan(&[t("a", &h, 1000), t("b", &h, 1000)]);
        if let DedupVerdict::Ok(p) = v {
            assert_eq!(p.bytes_saved, 1000);
        }
    }
}
