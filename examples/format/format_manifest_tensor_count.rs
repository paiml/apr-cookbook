//! # Format Manifest Tensor-Count Consistency
//!
//! Manifest declares N tensors; archive contains M tensor data blobs.
//! Mismatch indicates corruption (truncation), incomplete write
//! (M < N), or stale manifest (M > N). This recipe builds the
//! cross-checker + delta tier classifier.
//!
//! Demonstrates the **FMT.19** recipe for PMAT-130 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-FORMAT-001 §4 (manifest invariants).
//!
//! Run with: cargo run --example format_manifest_tensor_count
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ConsistencyVerdict {
    Ok,
    Truncated { manifest: u32, archive: u32 },
    StaleManifest { manifest: u32, archive: u32 },
    BothEmpty,
}

pub fn check(manifest_count: u32, archive_count: u32) -> ConsistencyVerdict {
    if manifest_count == 0 && archive_count == 0 {
        return ConsistencyVerdict::BothEmpty;
    }
    match manifest_count.cmp(&archive_count) {
        std::cmp::Ordering::Equal => ConsistencyVerdict::Ok,
        std::cmp::Ordering::Greater => ConsistencyVerdict::Truncated {
            manifest: manifest_count,
            archive: archive_count,
        },
        std::cmp::Ordering::Less => ConsistencyVerdict::StaleManifest {
            manifest: manifest_count,
            archive: archive_count,
        },
    }
}

#[derive(Debug, PartialEq)]
pub enum CorruptionTier {
    Pristine,
    MinorDelta { delta: u32 },
    SignificantDelta { delta: u32 },
    SevereCorruption { delta: u32 },
}

pub fn classify_corruption(manifest_count: u32, archive_count: u32) -> CorruptionTier {
    let delta = manifest_count.abs_diff(archive_count);
    if delta == 0 {
        CorruptionTier::Pristine
    } else if delta <= 2 {
        CorruptionTier::MinorDelta { delta }
    } else if delta <= 10 {
        CorruptionTier::SignificantDelta { delta }
    } else {
        CorruptionTier::SevereCorruption { delta }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_manifest_tensor_count")?;

    for (m, a) in [(100u32, 100u32), (100, 99), (100, 50), (50, 100), (0, 0)] {
        println!(
            "manifest={m} archive={a}  →  {:?}  corruption={:?}",
            check(m, a),
            classify_corruption(m, a)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_counts_ok() {
        assert_eq!(check(100, 100), ConsistencyVerdict::Ok);
    }

    #[test]
    fn manifest_larger_truncated() {
        let v = check(100, 99);
        assert!(matches!(
            v,
            ConsistencyVerdict::Truncated {
                manifest: 100,
                archive: 99
            }
        ));
    }

    #[test]
    fn archive_larger_stale_manifest() {
        let v = check(50, 100);
        assert!(matches!(
            v,
            ConsistencyVerdict::StaleManifest {
                manifest: 50,
                archive: 100
            }
        ));
    }

    #[test]
    fn both_empty_handled() {
        assert_eq!(check(0, 0), ConsistencyVerdict::BothEmpty);
    }

    #[test]
    fn corruption_pristine_at_zero_delta() {
        assert_eq!(classify_corruption(100, 100), CorruptionTier::Pristine);
    }

    #[test]
    fn corruption_minor_at_small_delta() {
        let v = classify_corruption(100, 99);
        assert!(matches!(v, CorruptionTier::MinorDelta { delta: 1 }));
    }

    #[test]
    fn corruption_significant_at_medium_delta() {
        let v = classify_corruption(100, 95);
        assert!(matches!(v, CorruptionTier::SignificantDelta { delta: 5 }));
    }

    #[test]
    fn corruption_severe_at_large_delta() {
        let v = classify_corruption(100, 50);
        assert!(matches!(v, CorruptionTier::SevereCorruption { delta: 50 }));
    }

    #[test]
    fn boundary_at_2_minor() {
        let v = classify_corruption(100, 98);
        assert!(matches!(v, CorruptionTier::MinorDelta { delta: 2 }));
    }

    #[test]
    fn boundary_at_3_significant() {
        let v = classify_corruption(100, 97);
        assert!(matches!(v, CorruptionTier::SignificantDelta { delta: 3 }));
    }

    #[test]
    fn delta_uses_abs_diff() {
        // archive > manifest also yields positive delta.
        let v = classify_corruption(50, 100);
        assert!(matches!(v, CorruptionTier::SevereCorruption { delta: 50 }));
    }
}
