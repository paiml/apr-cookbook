//! # Registry Per-User Storage Quota
//!
//! Tier limits:
//!   Free:      10 GiB, 100 models, 100 daily uploads
//!   Team:      100 GiB, 1k models, 1k daily uploads
//!   Enterprise: 10 TiB, 100k models, 100k daily uploads
//!
//! Pre-upload check: WouldExceed{kind} if any quota would be tripped.
//! This recipe builds the checker.
//!
//! Demonstrates the **REG.13** recipe for PMAT-138 (registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace Hub tier-storage limits.
//!
//! Run with: cargo run --example registry_quota_per_user
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Free,
    Team,
    Enterprise,
}

#[derive(Debug, Clone, Copy)]
pub struct Quota {
    pub max_storage_bytes: u64,
    pub max_models: u32,
    pub max_daily_uploads: u32,
}

impl Quota {
    pub fn for_tier(tier: Tier) -> Self {
        match tier {
            Tier::Free => Quota {
                max_storage_bytes: 10 * 1024 * 1024 * 1024,
                max_models: 100,
                max_daily_uploads: 100,
            },
            Tier::Team => Quota {
                max_storage_bytes: 100 * 1024 * 1024 * 1024,
                max_models: 1_000,
                max_daily_uploads: 1_000,
            },
            Tier::Enterprise => Quota {
                max_storage_bytes: 10u64 * 1024 * 1024 * 1024 * 1024,
                max_models: 100_000,
                max_daily_uploads: 100_000,
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum QuotaKind {
    Storage,
    ModelCount,
    DailyUploadCount,
}

#[derive(Debug, PartialEq)]
pub enum QuotaVerdict {
    Ok,
    WouldExceed { kind: QuotaKind, by: u64 },
    InvalidUploadSize,
}

pub fn check(
    tier: Tier,
    current_storage_bytes: u64,
    current_model_count: u32,
    today_upload_count: u32,
    upload_size_bytes: u64,
    is_new_model: bool,
) -> QuotaVerdict {
    if upload_size_bytes == 0 {
        return QuotaVerdict::InvalidUploadSize;
    }
    let q = Quota::for_tier(tier);
    let new_storage = current_storage_bytes + upload_size_bytes;
    if new_storage > q.max_storage_bytes {
        return QuotaVerdict::WouldExceed {
            kind: QuotaKind::Storage,
            by: new_storage - q.max_storage_bytes,
        };
    }
    if is_new_model && current_model_count >= q.max_models {
        return QuotaVerdict::WouldExceed {
            kind: QuotaKind::ModelCount,
            by: u64::from(current_model_count + 1 - q.max_models),
        };
    }
    if today_upload_count >= q.max_daily_uploads {
        return QuotaVerdict::WouldExceed {
            kind: QuotaKind::DailyUploadCount,
            by: u64::from(today_upload_count + 1 - q.max_daily_uploads),
        };
    }
    QuotaVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_quota_per_user")?;

    println!(
        "free, 5 GiB used, 1 GiB upload: {:?}",
        check(Tier::Free, 5u64 << 30, 50, 50, 1u64 << 30, true)
    );
    println!(
        "free, 10 GiB used, 1 GiB upload: {:?}",
        check(Tier::Free, 10u64 << 30, 50, 50, 1u64 << 30, true)
    );
    println!(
        "free, 100 models, new model: {:?}",
        check(Tier::Free, 1u64 << 30, 100, 50, 1024, true)
    );
    println!(
        "free, 100 daily uploads: {:?}",
        check(Tier::Free, 1u64 << 30, 50, 100, 1024, false)
    );
    println!(
        "invalid upload (0): {:?}",
        check(Tier::Free, 0, 0, 0, 0, true)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_all_limits_ok() {
        let v = check(Tier::Free, 1 << 30, 50, 50, 1 << 30, true);
        assert_eq!(v, QuotaVerdict::Ok);
    }

    #[test]
    fn storage_overflow_rejected() {
        let v = check(Tier::Free, 10u64 << 30, 50, 50, 1, true);
        assert!(matches!(
            v,
            QuotaVerdict::WouldExceed {
                kind: QuotaKind::Storage,
                ..
            }
        ));
    }

    #[test]
    fn model_count_overflow_rejected() {
        let v = check(Tier::Free, 1 << 30, 100, 50, 1024, true);
        assert!(matches!(
            v,
            QuotaVerdict::WouldExceed {
                kind: QuotaKind::ModelCount,
                ..
            }
        ));
    }

    #[test]
    fn updating_existing_model_does_not_count() {
        // is_new_model=false → model count check skipped.
        let v = check(Tier::Free, 1 << 30, 100, 50, 1024, false);
        assert_eq!(v, QuotaVerdict::Ok);
    }

    #[test]
    fn daily_upload_overflow_rejected() {
        let v = check(Tier::Free, 1 << 30, 50, 100, 1024, false);
        assert!(matches!(
            v,
            QuotaVerdict::WouldExceed {
                kind: QuotaKind::DailyUploadCount,
                ..
            }
        ));
    }

    #[test]
    fn zero_upload_invalid() {
        assert_eq!(
            check(Tier::Free, 0, 0, 0, 0, true),
            QuotaVerdict::InvalidUploadSize
        );
    }

    #[test]
    fn team_tier_higher_storage_limit() {
        // Same usage but Team tier passes where Free fails.
        let storage = 50u64 << 30; // 50 GiB
        let upload = 1u64 << 30;
        let team_v = check(Tier::Team, storage, 50, 50, upload, false);
        let free_v = check(Tier::Free, storage, 50, 50, upload, false);
        assert_eq!(team_v, QuotaVerdict::Ok);
        assert!(matches!(free_v, QuotaVerdict::WouldExceed { .. }));
    }

    #[test]
    fn enterprise_higher_model_count() {
        let v = check(Tier::Enterprise, 0, 50_000, 0, 1024, true);
        assert_eq!(v, QuotaVerdict::Ok);
    }

    #[test]
    fn excess_storage_amount_reported() {
        // 10 GiB Free quota, 10 GiB used, 1 byte upload → over by 1.
        let v = check(Tier::Free, 10u64 << 30, 0, 0, 1, true);
        if let QuotaVerdict::WouldExceed { by, .. } = v {
            assert_eq!(by, 1);
        }
    }

    #[test]
    fn quota_check_priority_order() {
        // Storage overflow checked before model count.
        let v = check(Tier::Free, 10u64 << 30, 100, 100, 1024, true);
        assert!(matches!(
            v,
            QuotaVerdict::WouldExceed {
                kind: QuotaKind::Storage,
                ..
            }
        ));
    }
}
