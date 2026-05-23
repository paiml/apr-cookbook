//! # Registry Per-Artifact Size Quota
//!
//! Single-blob registry artifacts have practical limits:
//!   < 5 GiB → InlineUpload (single PUT)
//!   5-50 GiB → MultipartUpload (chunked, resumable)
//!   ≥ 50 GiB → RequiresLfsMigration (Git LFS / S3-style external store)
//!
//! Demonstrates the **REG.21** recipe for PMAT-150 (registry round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub LFS + AWS S3 multipart upload thresholds.
//!
//! Run with: cargo run --example registry_artifact_size_quota
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadStrategy {
    InlineUpload,
    MultipartUpload,
    RequiresLfsMigration,
}

#[derive(Debug, PartialEq)]
pub enum SizeVerdict {
    Ok {
        strategy: UploadStrategy,
        chunks: u64,
    },
    InvalidSize,
}

const INLINE_LIMIT_GIB: u64 = 5;
const MULTIPART_LIMIT_GIB: u64 = 50;
const MULTIPART_CHUNK_GIB: u64 = 1;

pub fn check(size_bytes: u64) -> SizeVerdict {
    if size_bytes == 0 {
        return SizeVerdict::InvalidSize;
    }
    let gib = size_bytes / (1024 * 1024 * 1024);
    let strategy = if gib < INLINE_LIMIT_GIB {
        UploadStrategy::InlineUpload
    } else if gib < MULTIPART_LIMIT_GIB {
        UploadStrategy::MultipartUpload
    } else {
        UploadStrategy::RequiresLfsMigration
    };
    let chunks = match strategy {
        UploadStrategy::InlineUpload => 1,
        UploadStrategy::MultipartUpload => gib.div_ceil(MULTIPART_CHUNK_GIB),
        UploadStrategy::RequiresLfsMigration => 1,
    };
    SizeVerdict::Ok { strategy, chunks }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_artifact_size_quota")?;

    println!("100 MiB: {:?}", check(100 * 1024 * 1024));
    println!("10 GiB: {:?}", check(10u64 * 1024 * 1024 * 1024));
    println!("100 GiB: {:?}", check(100u64 * 1024 * 1024 * 1024));
    println!("invalid 0: {:?}", check(0));
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
    fn small_inline_upload() {
        let v = check(100 * 1024 * 1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::InlineUpload);
        }
    }

    #[test]
    fn medium_multipart() {
        let v = check(10u64 * 1024 * 1024 * 1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::MultipartUpload);
        }
    }

    #[test]
    fn large_lfs_migration() {
        let v = check(100u64 * 1024 * 1024 * 1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::RequiresLfsMigration);
        }
    }

    #[test]
    fn invalid_zero_size() {
        assert_eq!(check(0), SizeVerdict::InvalidSize);
    }

    #[test]
    fn inline_uses_one_chunk() {
        let v = check(100 * 1024 * 1024);
        if let SizeVerdict::Ok { chunks, .. } = v {
            assert_eq!(chunks, 1);
        }
    }

    #[test]
    fn multipart_chunks_proportional() {
        // 10 GiB / 1 GiB per chunk = 10 chunks.
        let v = check(10u64 * 1024 * 1024 * 1024);
        if let SizeVerdict::Ok { chunks, .. } = v {
            assert_eq!(chunks, 10);
        }
    }

    #[test]
    fn boundary_at_5gib_multipart() {
        let v = check(5u64 * 1024 * 1024 * 1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::MultipartUpload);
        }
    }

    #[test]
    fn just_below_5gib_inline() {
        // 5 GiB - 1 byte → still inline.
        let v = check(5u64 * 1024 * 1024 * 1024 - 1);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::InlineUpload);
        }
    }

    #[test]
    fn boundary_at_50gib_lfs() {
        let v = check(50u64 * 1024 * 1024 * 1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::RequiresLfsMigration);
        }
    }

    #[test]
    fn very_small_size_inline() {
        let v = check(1024);
        if let SizeVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, UploadStrategy::InlineUpload);
        }
    }
}
