//! # Monte-Carlo Throttle Token Bucket Fairness
//!
//! Sim N tenants sharing a single token-bucket throttle. Each tenant
//! has an independent burst rate. Reports per-tenant throughput and
//! fairness ratio (min/max).
//!
//! Demonstrates the **MC.94** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: stride scheduling (Waldspurger MIT 1995); Jain's
//!  fairness index (1984).
//!
//! Run with: cargo run --example mc_throttle_token_bucket_fairness
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FairnessVerdict {
    Ok {
        per_tenant_served: Vec<u32>,
        fairness_ratio: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    seconds: u32,
    capacity: u32,
    refill_per_sec: u32,
    tenant_burst_rates: &[u32],
) -> FairnessVerdict {
    if seconds == 0 || capacity == 0 || refill_per_sec == 0 || tenant_burst_rates.is_empty() {
        return FairnessVerdict::InvalidConfig;
    }
    if tenant_burst_rates.contains(&0) {
        return FairnessVerdict::InvalidConfig;
    }
    let n = tenant_burst_rates.len();
    let mut tokens: u32 = capacity;
    let mut served: Vec<u32> = vec![0; n];
    for _ in 0..seconds {
        tokens = (tokens + refill_per_sec).min(capacity);
        // Round-robin attempt to serve up to each tenant's burst rate.
        for (i, &rate) in tenant_burst_rates.iter().enumerate() {
            for _ in 0..rate {
                if tokens > 0 {
                    tokens -= 1;
                    served[i] += 1;
                } else {
                    break;
                }
            }
        }
    }
    let max_served = served.iter().max().copied().unwrap_or(1) as f64;
    let min_served = served.iter().min().copied().unwrap_or(0) as f64;
    let fairness_ratio = if max_served > 0.0 {
        min_served / max_served
    } else {
        1.0
    };
    FairnessVerdict::Ok {
        per_tenant_served: served,
        fairness_ratio,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_throttle_token_bucket_fairness")?;

    let rates = [10u32, 10, 10];
    println!("equal burst: {:?}", simulate(60, 100, 30, &rates));
    let uneven = [50u32, 5, 1];
    println!("uneven: {:?}", simulate(60, 100, 30, &uneven));
    println!("invalid: {:?}", simulate(0, 100, 30, &rates));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_seconds() {
        let rates = [10u32];
        assert_eq!(simulate(0, 100, 30, &rates), FairnessVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        let rates = [10u32];
        assert_eq!(simulate(60, 0, 30, &rates), FairnessVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_tenants() {
        assert_eq!(simulate(60, 100, 30, &[]), FairnessVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_tenant_rate() {
        let rates = [10u32, 0, 5];
        assert_eq!(
            simulate(60, 100, 30, &rates),
            FairnessVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let rates = [10u32, 10];
        let a = simulate(60, 100, 30, &rates);
        let b = simulate(60, 100, 30, &rates);
        assert_eq!(a, b);
    }

    #[test]
    fn per_tenant_count_matches_input() {
        let rates = [10u32, 10, 10, 10];
        let v = simulate(60, 100, 30, &rates);
        if let FairnessVerdict::Ok {
            per_tenant_served, ..
        } = v
        {
            assert_eq!(per_tenant_served.len(), 4);
        }
    }

    #[test]
    fn uneven_rates_uneven_serving() {
        let rates = [50u32, 5, 1];
        let v = simulate(60, 100, 30, &rates);
        if let FairnessVerdict::Ok { fairness_ratio, .. } = v {
            assert!(fairness_ratio < 1.0);
        }
    }

    #[test]
    fn equal_rates_high_fairness() {
        let rates = [5u32, 5, 5];
        let v = simulate(120, 100, 30, &rates);
        if let FairnessVerdict::Ok { fairness_ratio, .. } = v {
            assert!(fairness_ratio > 0.7);
        }
    }

    #[test]
    fn fairness_in_unit_range() {
        let rates = [5u32, 5];
        let v = simulate(60, 100, 30, &rates);
        if let FairnessVerdict::Ok { fairness_ratio, .. } = v {
            assert!((0.0..=1.0).contains(&fairness_ratio));
        }
    }

    #[test]
    fn served_total_le_refill_plus_initial() {
        let rates = [10u32, 10];
        let v = simulate(10, 100, 5, &rates);
        if let FairnessVerdict::Ok {
            per_tenant_served, ..
        } = v
        {
            let total: u32 = per_tenant_served.iter().sum();
            // initial 100 + 10 sec * 5 = 150 tokens max.
            assert!(total <= 150);
        }
    }

    #[test]
    fn single_tenant_full_service() {
        let rates = [10u32];
        let v = simulate(60, 100, 30, &rates);
        if let FairnessVerdict::Ok {
            fairness_ratio,
            per_tenant_served,
        } = v
        {
            // Single tenant → fairness_ratio always 1.
            assert_eq!(fairness_ratio, 1.0);
            assert!(per_tenant_served[0] > 0);
        }
    }
}
