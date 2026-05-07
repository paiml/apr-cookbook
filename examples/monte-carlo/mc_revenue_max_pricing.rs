//! # Monte-Carlo Revenue Maximization Pricing
//!
//! Sim revenue across discrete price points; demand falls linearly
//! with price. Find optimal price (revenue = price × demand).
//!
//! Demonstrates the **MC.131** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: linear demand curve (Marshall, Principles of Economics
//!  1890); revenue management.
//!
//! Run with: cargo run --example mc_revenue_max_pricing
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PricingVerdict {
    Ok {
        optimal_price: u32,
        max_revenue: u32,
        revenues: Vec<u32>,
    },
    InvalidConfig,
}

pub fn simulate(
    min_price: u32,
    max_price: u32,
    step: u32,
    demand_intercept: u32,
    demand_slope: u32,
) -> PricingVerdict {
    if min_price >= max_price || step == 0 || demand_slope == 0 {
        return PricingVerdict::InvalidConfig;
    }
    let mut revenues: Vec<u32> = Vec::new();
    let mut max_revenue = 0u32;
    let mut optimal_price = min_price;
    let mut price = min_price;
    while price <= max_price {
        // Linear demand: max(0, intercept - slope * price).
        let demand = demand_intercept.saturating_sub(demand_slope * price);
        let revenue = price * demand;
        revenues.push(revenue);
        if revenue > max_revenue {
            max_revenue = revenue;
            optimal_price = price;
        }
        price += step;
    }
    PricingVerdict::Ok {
        optimal_price,
        max_revenue,
        revenues,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_revenue_max_pricing")?;

    println!("typical: {:?}", simulate(1, 100, 5, 1000, 5));
    println!("invalid: {:?}", simulate(100, 50, 5, 1000, 5));
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
    fn invalid_min_ge_max() {
        assert_eq!(simulate(100, 50, 5, 1000, 5), PricingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_step() {
        assert_eq!(simulate(1, 100, 0, 1000, 5), PricingVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_slope() {
        assert_eq!(simulate(1, 100, 5, 1000, 0), PricingVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1, 100, 5, 1000, 5);
        let b = simulate(1, 100, 5, 1000, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn revenue_curve_has_interior_peak_for_wide_range() {
        // Wide range with high slope so peak is interior.
        let v = simulate(1, 200, 1, 1000, 10);
        if let PricingVerdict::Ok { revenues, .. } = v {
            let mut peak_idx = 0;
            let mut peak_rev = 0;
            for (i, r) in revenues.iter().enumerate() {
                if *r > peak_rev {
                    peak_rev = *r;
                    peak_idx = i;
                }
            }
            assert!(peak_idx > 0);
            assert!(peak_idx < revenues.len() - 1);
        }
    }

    #[test]
    fn optimal_in_range() {
        let v = simulate(1, 100, 5, 1000, 5);
        if let PricingVerdict::Ok { optimal_price, .. } = v {
            assert!((1..=100).contains(&optimal_price));
        }
    }

    #[test]
    fn max_revenue_positive() {
        let v = simulate(1, 100, 5, 1000, 5);
        if let PricingVerdict::Ok { max_revenue, .. } = v {
            assert!(max_revenue > 0);
        }
    }

    #[test]
    fn higher_intercept_higher_revenue() {
        let lo = simulate(1, 100, 5, 500, 5);
        let hi = simulate(1, 100, 5, 2000, 5);
        if let (
            PricingVerdict::Ok { max_revenue: l, .. },
            PricingVerdict::Ok { max_revenue: h, .. },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn revenues_count_matches_steps() {
        let v = simulate(1, 11, 1, 100, 1);
        if let PricingVerdict::Ok { revenues, .. } = v {
            assert_eq!(revenues.len(), 11);
        }
    }

    #[test]
    fn small_range_works() {
        let v = simulate(1, 5, 1, 100, 1);
        if let PricingVerdict::Ok { revenues, .. } = v {
            assert_eq!(revenues.len(), 5);
        }
    }

    #[test]
    fn high_price_zero_demand() {
        let v = simulate(900, 1000, 10, 1000, 5);
        if let PricingVerdict::Ok { revenues, .. } = v {
            // At price 900, demand = 1000 - 4500 → saturates to 0.
            for r in &revenues {
                assert_eq!(*r, 0);
            }
        }
    }
}
