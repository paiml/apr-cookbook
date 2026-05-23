//! # Recipe: Registry Quota Lint — Per-Tenant Overage Report
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr registry-quota-lint --observation-file observation.json` (tenant fail)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates per-tenant quota enforcement with three classes of finding:
//!  - **Hard error**: tenant exceeds its quota (must shed before next push)
//!  - **Warning**: tenant ≥ 90% of quota (operator should reach out)
//!  - **Info**: tenant ≥ 75% of quota (forecast-only)
//!
//! Distinct severities prevent the lint from spamming on healthy systems
//! while still surfacing the long-tail of accounts approaching the cap.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_quota_lint_tenant_overage
//! ```
//!
//! ## References
//! - aprender CRUX-A-22 (per-tenant quota observation).
//!
//! Added by PMAT-090 (expand-cookbooks followup — registry/cache lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantStatus {
    Healthy,
    InfoApproaching, // >= 75%
    WarnNearCap,     // >= 90%
    ErrorOverQuota,  // > 100%
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantFinding {
    pub id: String,
    pub status: TenantStatus,
    pub used_pct: u64, // integer percent for Eq friendliness
}

pub fn classify_tenants(obs: &Value) -> Vec<TenantFinding> {
    let mut out = Vec::new();
    let Some(arr) = obs.get("tenants").and_then(Value::as_array) else {
        return out;
    };
    for t in arr {
        let id = t
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let used = t.get("bytes_used").and_then(Value::as_u64).unwrap_or(0);
        let quota = t.get("bytes_quota").and_then(Value::as_u64).unwrap_or(1);
        let pct = (used * 100) / quota;
        let status = if used > quota {
            TenantStatus::ErrorOverQuota
        } else if pct >= 90 {
            TenantStatus::WarnNearCap
        } else if pct >= 75 {
            TenantStatus::InfoApproaching
        } else {
            TenantStatus::Healthy
        };
        out.push(TenantFinding {
            id,
            status,
            used_pct: pct,
        });
    }
    out
}

fn build_mixed_observation() -> Value {
    json!({
        "tenants": [
            { "id": "team-alpha", "bytes_used":  6_000_000_000u64, "bytes_quota": 10_000_000_000u64 }, // 60% Healthy
            { "id": "team-bravo", "bytes_used":  7_700_000_000u64, "bytes_quota": 10_000_000_000u64 }, // 77% Info
            { "id": "team-eval",  "bytes_used":  9_400_000_000u64, "bytes_quota": 10_000_000_000u64 }, // 94% Warn
            { "id": "team-runtime", "bytes_used": 12_000_000_000u64, "bytes_quota": 10_000_000_000u64 } // 120% Error
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_quota_lint_tenant_overage")?;
    let obs = build_mixed_observation();
    let findings = classify_tenants(&obs);

    println!("=== Recipe: {} ===", ctx.name());
    for f in &findings {
        println!("  {:>14} {}% → {:?}", f.id, f.used_pct, f.status);
    }
    let errors = findings
        .iter()
        .filter(|f| f.status == TenantStatus::ErrorOverQuota)
        .count();
    ctx.record_metric("error_tenants", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_overage_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn classifies_each_tenant_into_correct_band() {
        let f = classify_tenants(&build_mixed_observation());
        let states: Vec<TenantStatus> = f.into_iter().map(|x| x.status).collect();
        assert_eq!(
            states,
            vec![
                TenantStatus::Healthy,
                TenantStatus::InfoApproaching,
                TenantStatus::WarnNearCap,
                TenantStatus::ErrorOverQuota,
            ]
        );
    }

    #[test]
    fn empty_tenants_yields_no_findings() {
        let f = classify_tenants(&json!({ "tenants": [] }));
        assert!(f.is_empty());
    }

    #[test]
    fn missing_tenants_array_yields_no_findings() {
        let f = classify_tenants(&json!({}));
        assert!(f.is_empty());
    }

    #[test]
    fn boundary_at_exactly_90_pct_is_warn() {
        let obs = json!({
            "tenants": [{ "id": "t", "bytes_used": 90, "bytes_quota": 100 }]
        });
        let f = classify_tenants(&obs);
        assert_eq!(f[0].status, TenantStatus::WarnNearCap);
    }

    #[test]
    fn boundary_at_exactly_100_pct_is_warn_not_error() {
        // Equality with quota is permitted — only strict overage is error.
        let obs = json!({
            "tenants": [{ "id": "t", "bytes_used": 100, "bytes_quota": 100 }]
        });
        let f = classify_tenants(&obs);
        assert_eq!(f[0].status, TenantStatus::WarnNearCap);
    }
}
