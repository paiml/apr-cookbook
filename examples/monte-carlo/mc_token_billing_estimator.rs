//! # Monte-Carlo Token-Cost Billing Estimator
//!
//! Sim daily token costs under a variable workload. Tokens per request
//! drawn from a log-normal-ish distribution; cost = tokens × price.
//! Returns p50, p95 daily cost.
//!
//! Demonstrates the **MC.38** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OpenAI / Anthropic per-token pricing models.
//!
//! Run with: cargo run --example mc_token_billing_estimator
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BillingVerdict {
    Ok {
        p50_daily_cost_usd: f64,
        p95_daily_cost_usd: f64,
        max_daily_cost_usd: f64,
    },
    InvalidConfig,
}

pub fn estimate(
    requests_per_day_mean: u32,
    tokens_per_request_mean: u32,
    price_per_1k_tokens: f64,
    days: u32,
    seed: u64,
) -> BillingVerdict {
    if requests_per_day_mean == 0
        || tokens_per_request_mean == 0
        || days == 0
        || !price_per_1k_tokens.is_finite()
        || price_per_1k_tokens < 0.0
    {
        return BillingVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut daily_costs: Vec<f64> = Vec::with_capacity(days as usize);
    for _ in 0..days {
        // Daily request count in [0.5×, 1.5×] mean.
        let req_factor = 0.5 + unit(&mut rng_state);
        let requests = (f64::from(requests_per_day_mean) * req_factor) as u32;
        let mut day_tokens: u64 = 0;
        for _ in 0..requests {
            // Tokens per request in [0.5×, 2×] mean.
            let tok_factor = 0.5 + unit(&mut rng_state) * 1.5;
            let tokens = (f64::from(tokens_per_request_mean) * tok_factor) as u64;
            day_tokens += tokens;
        }
        let cost = (day_tokens as f64 / 1000.0) * price_per_1k_tokens;
        daily_costs.push(cost);
    }
    daily_costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = daily_costs.len();
    let p50 = daily_costs[(n as f64 * 0.50) as usize];
    let p95 = daily_costs[((n as f64 * 0.95) as usize).min(n - 1)];
    let max = *daily_costs.last().unwrap_or(&0.0);
    BillingVerdict::Ok {
        p50_daily_cost_usd: p50,
        p95_daily_cost_usd: p95,
        max_daily_cost_usd: max,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_billing_estimator")?;

    println!("typical: {:?}", estimate(10_000, 500, 0.002, 30, 42));
    println!("small: {:?}", estimate(100, 200, 0.002, 7, 42));
    println!("invalid: {:?}", estimate(0, 100, 0.002, 30, 42));
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
    fn p95_above_p50() {
        let v = estimate(10_000, 500, 0.002, 30, 42);
        if let BillingVerdict::Ok {
            p50_daily_cost_usd,
            p95_daily_cost_usd,
            ..
        } = v
        {
            assert!(p95_daily_cost_usd >= p50_daily_cost_usd);
        }
    }

    #[test]
    fn max_above_p95() {
        let v = estimate(10_000, 500, 0.002, 30, 42);
        if let BillingVerdict::Ok {
            p95_daily_cost_usd,
            max_daily_cost_usd,
            ..
        } = v
        {
            assert!(max_daily_cost_usd >= p95_daily_cost_usd);
        }
    }

    #[test]
    fn higher_price_higher_cost() {
        let v_low = estimate(1000, 500, 0.001, 30, 42);
        let v_high = estimate(1000, 500, 0.01, 30, 42);
        if let (
            BillingVerdict::Ok {
                p50_daily_cost_usd: l,
                ..
            },
            BillingVerdict::Ok {
                p50_daily_cost_usd: h,
                ..
            },
        ) = (v_low, v_high)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(
            estimate(0, 100, 0.002, 30, 42),
            BillingVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_tokens() {
        assert_eq!(
            estimate(100, 0, 0.002, 30, 42),
            BillingVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_days() {
        assert_eq!(
            estimate(100, 500, 0.002, 0, 42),
            BillingVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_price() {
        assert_eq!(
            estimate(100, 500, -0.001, 30, 42),
            BillingVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            estimate(100, 500, f64::NAN, 30, 42),
            BillingVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = estimate(100, 500, 0.002, 7, 42);
        let b = estimate(100, 500, 0.002, 7, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn cost_non_negative() {
        let v = estimate(100, 500, 0.002, 7, 42);
        if let BillingVerdict::Ok {
            p50_daily_cost_usd, ..
        } = v
        {
            assert!(p50_daily_cost_usd >= 0.0);
        }
    }
}
