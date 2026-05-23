//! # Recipe: RM with Age-Based Retention Policy
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr rm --retention-days 30 --keep-latest 2`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example rm_retention_policy` exits 0
//! 2. [x] `cargo test --example rm_retention_policy` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr rm --retention-days --keep-latest` in-process
//! 10. [x] Unit tests cover retention window, keep-latest, boundary conditions
//!
//! ## Learning Objective
//! Demonstrates age-based cache eviction with a "keep at least N latest"
//! override. Entries older than the retention window are candidates for
//! deletion, but the policy always preserves the N most-recent entries to
//! guarantee the cache is never empty.
//!
//! ## Run Command
//! ```bash
//! cargo run --example rm_retention_policy
//! ```
//!
//! ## References
//! - Karger, D. et al. (1997). *Consistent Hashing and Random Trees*. STOC. DOI: 10.1145/258533.258660

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    pub name: String,
    pub last_access_days_ago: u32,
    pub size_bytes: u64,
}

/// Apply a retention policy: evict entries older than `retention_days`,
/// but always keep the `keep_latest` most-recent (smallest `last_access_days_ago`).
pub fn apply_retention(
    entries: &[CachedModel],
    retention_days: u32,
    keep_latest: usize,
) -> (Vec<CachedModel>, Vec<CachedModel>) {
    let mut sorted: Vec<CachedModel> = entries.to_vec();
    // Smallest age first (most recent).
    sorted.sort_by(|a, b| {
        a.last_access_days_ago
            .cmp(&b.last_access_days_ago)
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut kept = Vec::new();
    let mut evicted = Vec::new();

    for (idx, e) in sorted.iter().enumerate() {
        let is_in_keep_latest = idx < keep_latest;
        let is_within_retention = e.last_access_days_ago <= retention_days;
        if is_in_keep_latest || is_within_retention {
            kept.push(e.clone());
        } else {
            evicted.push(e.clone());
        }
    }

    (kept, evicted)
}

pub fn fixture() -> Vec<CachedModel> {
    vec![
        CachedModel {
            name: "phi-3-mini".into(),
            last_access_days_ago: 2,
            size_bytes: 3_800_000_000,
        },
        CachedModel {
            name: "llama-3.2-1b".into(),
            last_access_days_ago: 7,
            size_bytes: 1_100_000_000,
        },
        CachedModel {
            name: "bert-base".into(),
            last_access_days_ago: 35,
            size_bytes: 440_000_000,
        },
        CachedModel {
            name: "whisper-tiny".into(),
            last_access_days_ago: 120,
            size_bytes: 75_000_000,
        },
        CachedModel {
            name: "distilgpt2".into(),
            last_access_days_ago: 95,
            size_bytes: 320_000_000,
        },
    ]
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("rm_retention_policy")?;
    println!("=== Recipe: {} ===", ctx.name());

    let retention_days = 30u32;
    let keep_latest = 2usize;
    let all = fixture();
    let (kept, evicted) = apply_retention(&all, retention_days, keep_latest);

    println!(
        "Policy: retain ≤ {} days, keep latest {}",
        retention_days, keep_latest
    );
    println!("Total entries:  {}", all.len());
    println!("Kept:           {}", kept.len());
    println!("Evicted:        {}", evicted.len());
    for e in &evicted {
        println!(
            "  - {} ({}d old, {} bytes)",
            e.name, e.last_access_days_ago, e.size_bytes
        );
    }

    let bytes_freed: u64 = evicted.iter().map(|e| e.size_bytes).sum();
    let report = json!({
        "recipe": ctx.name(),
        "retention_days": retention_days,
        "keep_latest": keep_latest,
        "n_total": all.len(),
        "n_kept": kept.len(),
        "n_evicted": evicted.len(),
        "bytes_freed": bytes_freed,
        "evicted": evicted.iter().map(|e| json!({
            "name": e.name,
            "age_days": e.last_access_days_ago,
            "size_bytes": e.size_bytes,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("rm-retention.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_entries_older_than_window() {
        let (_, evicted) = apply_retention(&fixture(), 30, 0);
        assert!(evicted.iter().all(|e| e.last_access_days_ago > 30));
        assert_eq!(evicted.len(), 3);
    }

    #[test]
    fn keep_latest_overrides_retention() {
        // Keep the 5 newest even if they'd be evicted.
        let (kept, evicted) = apply_retention(&fixture(), 0, 5);
        assert_eq!(kept.len(), 5);
        assert_eq!(evicted.len(), 0);
    }

    #[test]
    fn zero_retention_zero_keep_evicts_all_but_within_zero() {
        // retention_days=0 means "evict if > 0" so 0-day old entries stay.
        let (_, evicted) = apply_retention(&fixture(), 0, 0);
        // No entries in fixture have age 0, so ALL should be evicted.
        assert_eq!(evicted.len(), 5);
    }

    #[test]
    fn policy_is_idempotent() {
        let p1 = apply_retention(&fixture(), 30, 2);
        let p2 = apply_retention(&fixture(), 30, 2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn boundary_at_retention_day_is_kept() {
        let e = vec![CachedModel {
            name: "edge".into(),
            last_access_days_ago: 30,
            size_bytes: 1,
        }];
        let (kept, evicted) = apply_retention(&e, 30, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(evicted.len(), 0);
    }
}
