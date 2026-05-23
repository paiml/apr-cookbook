//! # Advanced Per-Tenant Admission Quota
//!
//! Multi-tenant inference: each tenant has a daily request budget.
//! Track count, reject when over.
//!
//! Demonstrates the **ADV.37** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS API Gateway usage plans (per-API-key quota).
//!
//! Run with: cargo run --example adv_admission_quota_per_tenant
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuotaVerdict {
    Admit {
        tenant_remaining: u32,
        global_remaining: u32,
    },
    TenantQuotaExceeded {
        used: u32,
        limit: u32,
    },
    GlobalQuotaExceeded,
    InvalidConfig,
}

pub fn check(
    tenant_used: u32,
    tenant_limit: u32,
    global_used: u32,
    global_limit: u32,
) -> QuotaVerdict {
    if tenant_limit == 0 || global_limit == 0 {
        return QuotaVerdict::InvalidConfig;
    }
    if tenant_used >= tenant_limit {
        return QuotaVerdict::TenantQuotaExceeded {
            used: tenant_used,
            limit: tenant_limit,
        };
    }
    if global_used >= global_limit {
        return QuotaVerdict::GlobalQuotaExceeded;
    }
    QuotaVerdict::Admit {
        tenant_remaining: tenant_limit - tenant_used - 1,
        global_remaining: global_limit - global_used - 1,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_admission_quota_per_tenant")?;

    println!("ok: {:?}", check(50, 1000, 5_000, 100_000));
    println!("tenant exceeded: {:?}", check(1000, 1000, 5_000, 100_000));
    println!("global exceeded: {:?}", check(50, 1000, 100_000, 100_000));
    println!("invalid: {:?}", check(50, 0, 5_000, 100_000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn admitted_within_quota() {
        let v = check(50, 1000, 5_000, 100_000);
        assert!(matches!(v, QuotaVerdict::Admit { .. }));
    }

    #[test]
    fn tenant_at_limit_rejected() {
        let v = check(1000, 1000, 5_000, 100_000);
        assert!(matches!(v, QuotaVerdict::TenantQuotaExceeded { .. }));
    }

    #[test]
    fn tenant_over_limit_rejected() {
        let v = check(2000, 1000, 5_000, 100_000);
        assert!(matches!(v, QuotaVerdict::TenantQuotaExceeded { .. }));
    }

    #[test]
    fn global_at_limit_rejected() {
        let v = check(50, 1000, 100_000, 100_000);
        assert_eq!(v, QuotaVerdict::GlobalQuotaExceeded);
    }

    #[test]
    fn tenant_takes_precedence_over_global() {
        let v = check(1000, 1000, 100_000, 100_000);
        assert!(matches!(v, QuotaVerdict::TenantQuotaExceeded { .. }));
    }

    #[test]
    fn zero_tenant_limit_invalid() {
        assert_eq!(check(50, 0, 5_000, 100_000), QuotaVerdict::InvalidConfig);
    }

    #[test]
    fn zero_global_limit_invalid() {
        assert_eq!(check(50, 1000, 5_000, 0), QuotaVerdict::InvalidConfig);
    }

    #[test]
    fn remaining_count_correct() {
        let v = check(50, 1000, 5_000, 100_000);
        if let QuotaVerdict::Admit {
            tenant_remaining,
            global_remaining,
        } = v
        {
            assert_eq!(tenant_remaining, 949);
            assert_eq!(global_remaining, 94_999);
        }
    }

    #[test]
    fn last_admit_just_before_limit() {
        let v = check(999, 1000, 5_000, 100_000);
        if let QuotaVerdict::Admit {
            tenant_remaining, ..
        } = v
        {
            assert_eq!(tenant_remaining, 0);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(50, 1000, 5_000, 100_000);
        let b = check(50, 1000, 5_000, 100_000);
        assert_eq!(a, b);
    }
}
