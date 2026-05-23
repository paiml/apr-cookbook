//! # Advanced Token Budget Estimator
//!
//! Estimate cost: tokens × $/1k_token. Prompt + completion tokens
//! priced separately on most providers.
//!
//! Plus tier classification: under_dollar (< $1), normal (< $10),
//! expensive (< $100), critical (≥ $100 — needs approval).
//!
//! Demonstrates the **ADV.21** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenAI/Anthropic published per-1k-token pricing.
//!
//! Run with: cargo run --example adv_token_budget
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostTier {
    UnderDollar,
    Normal,
    Expensive,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok { cost_usd: f64, tier: CostTier },
    InvalidPricing,
    InvalidTokens,
}

pub fn estimate(
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_per_1k: f64,
    completion_per_1k: f64,
) -> BudgetVerdict {
    if !prompt_per_1k.is_finite()
        || !completion_per_1k.is_finite()
        || prompt_per_1k < 0.0
        || completion_per_1k < 0.0
    {
        return BudgetVerdict::InvalidPricing;
    }
    if prompt_tokens == 0 && completion_tokens == 0 {
        return BudgetVerdict::InvalidTokens;
    }
    let cost_usd = (f64::from(prompt_tokens) * prompt_per_1k
        + f64::from(completion_tokens) * completion_per_1k)
        / 1000.0;
    let tier = if cost_usd < 1.0 {
        CostTier::UnderDollar
    } else if cost_usd < 10.0 {
        CostTier::Normal
    } else if cost_usd < 100.0 {
        CostTier::Expensive
    } else {
        CostTier::Critical
    };
    BudgetVerdict::Ok { cost_usd, tier }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_token_budget")?;

    println!("100 in / 100 out: {:?}", estimate(100, 100, 0.001, 0.003));
    println!(
        "10k in / 5k out: {:?}",
        estimate(10_000, 5_000, 0.001, 0.003)
    );
    println!(
        "100k in / 50k out: {:?}",
        estimate(100_000, 50_000, 0.001, 0.003)
    );
    println!(
        "1M in / 500k out: {:?}",
        estimate(1_000_000, 500_000, 0.001, 0.003)
    );
    println!("invalid: {:?}", estimate(0, 0, 0.001, 0.003));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_under_dollar() {
        let v = estimate(100, 100, 0.001, 0.003);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::UnderDollar);
        }
    }

    #[test]
    fn medium_normal() {
        // 10k × 0.001 + 5k × 0.003 = 10 + 15 = 25? No: per-1k.
        // 10k tok × $0.001/1k = $0.01 + 5k × $0.003/1k = $0.015 = $0.025.
        // That's under_dollar. Need bigger to hit normal.
        let v = estimate(1_000_000, 500_000, 0.001, 0.003);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Normal);
        }
    }

    #[test]
    fn large_expensive() {
        // 10M tokens at $0.001/1k = $10.
        let v = estimate(10_000_000, 0, 0.001, 0.001);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Expensive);
        }
    }

    #[test]
    fn excessive_critical() {
        let v = estimate(100_000_000, 0, 0.001, 0.001);
        if let BudgetVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CostTier::Critical);
        }
    }

    #[test]
    fn invalid_negative_pricing() {
        assert_eq!(
            estimate(100, 100, -0.001, 0.001),
            BudgetVerdict::InvalidPricing
        );
    }

    #[test]
    fn invalid_zero_tokens() {
        assert_eq!(estimate(0, 0, 0.001, 0.001), BudgetVerdict::InvalidTokens);
    }

    #[test]
    fn cost_proportional_to_tokens() {
        let v_1k = estimate(1_000, 0, 0.001, 0.001);
        let v_2k = estimate(2_000, 0, 0.001, 0.001);
        if let (BudgetVerdict::Ok { cost_usd: c1, .. }, BudgetVerdict::Ok { cost_usd: c2, .. }) =
            (v_1k, v_2k)
        {
            assert!((c2 / c1 - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn completion_more_expensive_than_prompt() {
        let v_prompt = estimate(1_000_000, 0, 0.001, 0.003);
        let v_completion = estimate(0, 1_000_000, 0.001, 0.003);
        if let (BudgetVerdict::Ok { cost_usd: cp, .. }, BudgetVerdict::Ok { cost_usd: cc, .. }) =
            (v_prompt, v_completion)
        {
            assert!(cc > cp);
        }
    }

    #[test]
    fn nan_pricing_invalid() {
        assert_eq!(
            estimate(100, 100, f64::NAN, 0.001),
            BudgetVerdict::InvalidPricing
        );
    }

    #[test]
    fn cost_floor_zero_for_zero_pricing() {
        let v = estimate(1000, 1000, 0.0, 0.0);
        if let BudgetVerdict::Ok { cost_usd, .. } = v {
            assert!(cost_usd.abs() < 1e-9);
        }
    }
}
