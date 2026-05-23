//! # Serverless Lambda Pricing Matrix
//!
//! AWS Lambda pricing (us-east-1, 2024): $0.0000166667 per GB-second
//! plus $0.20 per 1M requests. Cost(req, dur, mem_mb) = req × $0.20/1M
//! plus req × dur × (mem_mb/1024) × $0.0000166667. This recipe builds
//! the calculator and the free-tier deduction.
//!
//! Demonstrates the **SVL.6** recipe for PMAT-126 (serverless coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda Pricing (us-east-1).
//!
//! Run with: cargo run --example serverless_lambda_pricing_matrix
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PER_REQUEST_USD: f64 = 0.20 / 1_000_000.0;
const PER_GB_SECOND_USD: f64 = 0.000_016_666_7;
const FREE_TIER_REQUESTS: u64 = 1_000_000;
const FREE_TIER_GB_SECONDS: f64 = 400_000.0;

#[derive(Debug, PartialEq)]
pub enum CostVerdict {
    Ok { cost_usd: f64 },
    InvalidMemory,
    InvalidDuration,
}

pub fn compute_cost(num_requests: u64, avg_duration_ms: u64, mem_mb: u32) -> CostVerdict {
    if mem_mb == 0 {
        return CostVerdict::InvalidMemory;
    }
    if avg_duration_ms == 0 {
        return CostVerdict::InvalidDuration;
    }
    let request_cost = num_requests as f64 * PER_REQUEST_USD;
    let mem_gb = f64::from(mem_mb) / 1024.0;
    let dur_secs = avg_duration_ms as f64 / 1000.0;
    let gb_secs = num_requests as f64 * dur_secs * mem_gb;
    let compute_cost = gb_secs * PER_GB_SECOND_USD;
    CostVerdict::Ok {
        cost_usd: request_cost + compute_cost,
    }
}

#[derive(Debug)]
pub struct FreeTierApplied {
    pub gross_usd: f64,
    pub net_usd: f64,
    pub requests_remaining: u64,
    pub gb_secs_remaining: f64,
}

pub fn apply_free_tier(
    num_requests: u64,
    avg_duration_ms: u64,
    mem_mb: u32,
) -> Option<FreeTierApplied> {
    if mem_mb == 0 || avg_duration_ms == 0 {
        return None;
    }
    let CostVerdict::Ok { cost_usd: gross } = compute_cost(num_requests, avg_duration_ms, mem_mb)
    else {
        return None;
    };
    let billable_requests = num_requests.saturating_sub(FREE_TIER_REQUESTS);
    let mem_gb = f64::from(mem_mb) / 1024.0;
    let dur_secs = avg_duration_ms as f64 / 1000.0;
    let gb_secs_total = num_requests as f64 * dur_secs * mem_gb;
    let billable_gb_secs = (gb_secs_total - FREE_TIER_GB_SECONDS).max(0.0);
    let net = billable_requests as f64 * PER_REQUEST_USD + billable_gb_secs * PER_GB_SECOND_USD;
    let req_remaining = FREE_TIER_REQUESTS.saturating_sub(num_requests);
    let gbs_remaining = (FREE_TIER_GB_SECONDS - gb_secs_total).max(0.0);
    Some(FreeTierApplied {
        gross_usd: gross,
        net_usd: net,
        requests_remaining: req_remaining,
        gb_secs_remaining: gbs_remaining,
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_lambda_pricing_matrix")?;

    let cases = [
        (1_000_000u64, 100u64, 256u32),
        (10_000_000, 500, 1024),
        (100, 1000, 0),
    ];
    for (req, dur, mem) in cases {
        println!(
            "req={req} dur={dur}ms mem={mem}MB  →  {:?}",
            compute_cost(req, dur, mem)
        );
    }
    let ft = apply_free_tier(2_000_000, 100, 256).unwrap();
    println!(
        "free-tier: gross=${:.4} net=${:.4} req_left={} gbs_left={:.0}",
        ft.gross_usd, ft.net_usd, ft.requests_remaining, ft.gb_secs_remaining
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_cost_calc() {
        // 1M req × 100ms × 256MB.
        let v = compute_cost(1_000_000, 100, 256);
        assert!(matches!(v, CostVerdict::Ok { .. }));
    }

    #[test]
    fn zero_memory_invalid() {
        assert_eq!(compute_cost(100, 100, 0), CostVerdict::InvalidMemory);
    }

    #[test]
    fn zero_duration_invalid() {
        assert_eq!(compute_cost(100, 0, 256), CostVerdict::InvalidDuration);
    }

    #[test]
    fn cost_scales_with_requests() {
        let a = compute_cost(1000, 100, 256);
        let b = compute_cost(2000, 100, 256);
        if let (CostVerdict::Ok { cost_usd: cost_a }, CostVerdict::Ok { cost_usd: cost_b }) = (a, b)
        {
            assert!((cost_b - 2.0 * cost_a).abs() < 1e-9);
        }
    }

    #[test]
    fn cost_scales_with_memory() {
        let a = compute_cost(1000, 100, 128);
        let b = compute_cost(1000, 100, 256);
        if let (CostVerdict::Ok { cost_usd: cost_a }, CostVerdict::Ok { cost_usd: cost_b }) = (a, b)
        {
            // Doubling memory increases compute cost (request cost stays same).
            assert!(cost_b > cost_a);
        }
    }

    #[test]
    fn cost_scales_with_duration() {
        let a = compute_cost(1000, 100, 256);
        let b = compute_cost(1000, 200, 256);
        if let (CostVerdict::Ok { cost_usd: cost_a }, CostVerdict::Ok { cost_usd: cost_b }) = (a, b)
        {
            assert!(cost_b > cost_a);
        }
    }

    #[test]
    fn free_tier_zeros_net_under_threshold() {
        // 100K reqs × 100ms × 128MB — well under free tier.
        let ft = apply_free_tier(100_000, 100, 128).unwrap();
        assert!(ft.net_usd < 1e-6);
        assert!(ft.requests_remaining > 0);
    }

    #[test]
    fn free_tier_partial_charge_above_threshold() {
        // 2M reqs (1M over free) × 100ms × 256MB.
        let ft = apply_free_tier(2_000_000, 100, 256).unwrap();
        assert!(ft.net_usd > 0.0);
        assert!(ft.net_usd < ft.gross_usd);
        assert_eq!(ft.requests_remaining, 0);
    }

    #[test]
    fn free_tier_invalid_inputs_return_none() {
        assert!(apply_free_tier(100, 0, 256).is_none());
        assert!(apply_free_tier(100, 100, 0).is_none());
    }
}
