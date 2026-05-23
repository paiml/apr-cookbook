//! # Serverless Account-Concurrency Limit Picker
//!
//! AWS Lambda account-level concurrency = 1000 (default; can be raised).
//! Reserved concurrency carves out a guaranteed pool for one function.
//! Provisioned concurrency = always-warm instances.
//!
//! Picker: given (function_p99_qps, total_account_qps, account_limit),
//! returns recommended_reserved + recommended_provisioned + tier.
//!
//! Demonstrates the **SVL.12** recipe for PMAT-144 (serverless round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda concurrency model docs.
//!
//! Run with: cargo run --example serverless_concurrency_limit
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyTier {
    Comfortable,
    AtLimit,
    NeedsRaise,
}

#[derive(Debug, PartialEq)]
pub enum LimitVerdict {
    Ok {
        recommended_reserved: u32,
        recommended_provisioned: u32,
        tier: ConcurrencyTier,
    },
    InvalidAccountLimit,
    InvalidFunctionDemand,
}

const HEADROOM_FACTOR: f64 = 1.20;

pub fn pick(
    function_p99_concurrent: u32,
    other_functions_concurrent: u32,
    account_limit: u32,
) -> LimitVerdict {
    if account_limit == 0 {
        return LimitVerdict::InvalidAccountLimit;
    }
    if function_p99_concurrent == 0 {
        return LimitVerdict::InvalidFunctionDemand;
    }
    let total_demand = function_p99_concurrent.saturating_add(other_functions_concurrent);
    let with_headroom = (f64::from(total_demand) * HEADROOM_FACTOR) as u32;
    let tier = if with_headroom > account_limit {
        ConcurrencyTier::NeedsRaise
    } else if with_headroom > account_limit * 8 / 10 {
        ConcurrencyTier::AtLimit
    } else {
        ConcurrencyTier::Comfortable
    };
    let recommended_reserved = (f64::from(function_p99_concurrent) * HEADROOM_FACTOR) as u32;
    // Provision base load; remainder scales on demand.
    let recommended_provisioned = function_p99_concurrent / 2;
    LimitVerdict::Ok {
        recommended_reserved,
        recommended_provisioned,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_concurrency_limit")?;

    println!("comfortable: {:?}", pick(100, 200, 1000));
    println!("at limit: {:?}", pick(500, 250, 1000));
    println!("needs raise: {:?}", pick(800, 500, 1000));
    println!("invalid limit: {:?}", pick(100, 0, 0));
    println!("invalid zero demand: {:?}", pick(0, 0, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn comfortable_when_under_80_pct() {
        let v = pick(100, 200, 1000);
        if let LimitVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ConcurrencyTier::Comfortable);
        }
    }

    #[test]
    fn at_limit_above_80_pct() {
        let v = pick(500, 250, 1000);
        if let LimitVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ConcurrencyTier::AtLimit);
        }
    }

    #[test]
    fn needs_raise_above_account_limit() {
        let v = pick(800, 500, 1000);
        if let LimitVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ConcurrencyTier::NeedsRaise);
        }
    }

    #[test]
    fn invalid_zero_account_limit() {
        assert_eq!(pick(100, 0, 0), LimitVerdict::InvalidAccountLimit);
    }

    #[test]
    fn invalid_zero_function_demand() {
        assert_eq!(pick(0, 100, 1000), LimitVerdict::InvalidFunctionDemand);
    }

    #[test]
    fn reserved_includes_headroom() {
        // recommended_reserved = 100 × 1.2 = 120.
        let v = pick(100, 0, 1000);
        if let LimitVerdict::Ok {
            recommended_reserved,
            ..
        } = v
        {
            assert_eq!(recommended_reserved, 120);
        }
    }

    #[test]
    fn provisioned_half_of_p99() {
        let v = pick(100, 0, 1000);
        if let LimitVerdict::Ok {
            recommended_provisioned,
            ..
        } = v
        {
            assert_eq!(recommended_provisioned, 50);
        }
    }

    #[test]
    fn larger_function_more_reserved() {
        let v_small = pick(100, 0, 10_000);
        let v_large = pick(1_000, 0, 10_000);
        if let (
            LimitVerdict::Ok {
                recommended_reserved: s,
                ..
            },
            LimitVerdict::Ok {
                recommended_reserved: l,
                ..
            },
        ) = (v_small, v_large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn other_functions_count_toward_account_total() {
        let v_alone = pick(500, 0, 1000);
        let v_with_others = pick(500, 400, 1000);
        if let (LimitVerdict::Ok { tier: a, .. }, LimitVerdict::Ok { tier: b, .. }) =
            (v_alone, v_with_others)
        {
            // Tier should worsen when other functions consume capacity.
            assert!(a == ConcurrencyTier::Comfortable && b != ConcurrencyTier::Comfortable);
        }
    }

    #[test]
    fn at_account_limit_exactly_needs_raise() {
        // Total = 800, with headroom = 960; account = 800 → needs raise (>account).
        let v = pick(500, 300, 800);
        if let LimitVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, ConcurrencyTier::NeedsRaise);
        }
    }
}
