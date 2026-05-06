//! # Monte-Carlo Token Pricing Arbitrage
//!
//! Sim N exchanges with random price spreads. Detect arbitrage
//! opportunities (max - min > threshold). Reports opportunity rate
//! per tick.
//!
//! Demonstrates the **MC.106** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: triangular arbitrage in FX markets (Krugman 1981);
//!  CEX/DEX price-spread literature.
//!
//! Run with: cargo run --example mc_token_pricing_arbitrage
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ArbitrageVerdict {
    Ok {
        opportunity_rate: f64,
        max_spread: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    ticks: u32,
    exchanges: u32,
    base_price: f64,
    spread_pct: f64,
    threshold_pct: f64,
    seed: u64,
) -> ArbitrageVerdict {
    if ticks == 0 || exchanges < 2 || base_price <= 0.0 || spread_pct < 0.0 || threshold_pct < 0.0 {
        return ArbitrageVerdict::InvalidConfig;
    }
    let mut opportunities = 0u32;
    let mut max_spread = 0.0;
    let mut rng_state = seed | 1;
    for _ in 0..ticks {
        let mut prices: Vec<f64> = Vec::with_capacity(exchanges as usize);
        for _ in 0..exchanges {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64) - 0.5;
            let price = base_price * (1.0 + r * spread_pct / 100.0);
            prices.push(price);
        }
        let max = prices.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min = prices.iter().fold(f64::MAX, |a, &b| a.min(b));
        let spread_observed = (max - min) / base_price * 100.0;
        if spread_observed > max_spread {
            max_spread = spread_observed;
        }
        if spread_observed > threshold_pct {
            opportunities += 1;
        }
    }
    ArbitrageVerdict::Ok {
        opportunity_rate: f64::from(opportunities) / f64::from(ticks),
        max_spread,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_pricing_arbitrage")?;

    println!("tight market: {:?}", simulate(2000, 5, 100.0, 0.5, 0.3, 42));
    println!("wide spreads: {:?}", simulate(2000, 5, 100.0, 5.0, 1.0, 42));
    println!("invalid: {:?}", simulate(0, 5, 100.0, 1.0, 0.5, 42));
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
    fn wide_spreads_high_opportunities() {
        let v = simulate(2000, 5, 100.0, 5.0, 0.5, 42);
        if let ArbitrageVerdict::Ok {
            opportunity_rate, ..
        } = v
        {
            assert!(opportunity_rate > 0.5);
        }
    }

    #[test]
    fn tight_market_few_opportunities() {
        let v = simulate(2000, 5, 100.0, 0.1, 1.0, 42);
        if let ArbitrageVerdict::Ok {
            opportunity_rate, ..
        } = v
        {
            assert!(opportunity_rate < 0.05);
        }
    }

    #[test]
    fn invalid_zero_ticks() {
        assert_eq!(
            simulate(0, 5, 100.0, 1.0, 0.5, 42),
            ArbitrageVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_one_exchange() {
        assert_eq!(
            simulate(100, 1, 100.0, 1.0, 0.5, 42),
            ArbitrageVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_price() {
        assert_eq!(
            simulate(100, 5, 0.0, 1.0, 0.5, 42),
            ArbitrageVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 100.0, 1.0, 0.5, 42);
        let b = simulate(500, 5, 100.0, 1.0, 0.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(500, 5, 100.0, 1.0, 0.5, 42);
        if let ArbitrageVerdict::Ok {
            opportunity_rate, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&opportunity_rate));
        }
    }

    #[test]
    fn max_spread_finite() {
        let v = simulate(500, 5, 100.0, 1.0, 0.5, 42);
        if let ArbitrageVerdict::Ok { max_spread, .. } = v {
            assert!(max_spread.is_finite());
        }
    }

    #[test]
    fn higher_spread_higher_max_observed() {
        let lo = simulate(2000, 5, 100.0, 0.1, 0.0, 42);
        let hi = simulate(2000, 5, 100.0, 5.0, 0.0, 42);
        if let (
            ArbitrageVerdict::Ok { max_spread: l, .. },
            ArbitrageVerdict::Ok { max_spread: h, .. },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn zero_spread_zero_opportunities() {
        let v = simulate(100, 5, 100.0, 0.0, 0.5, 42);
        if let ArbitrageVerdict::Ok {
            opportunity_rate, ..
        } = v
        {
            assert_eq!(opportunity_rate, 0.0);
        }
    }

    #[test]
    fn more_exchanges_more_spread_potential() {
        let two = simulate(2000, 2, 100.0, 5.0, 0.0, 42);
        let many = simulate(2000, 10, 100.0, 5.0, 0.0, 42);
        if let (
            ArbitrageVerdict::Ok { max_spread: t, .. },
            ArbitrageVerdict::Ok { max_spread: m, .. },
        ) = (two, many)
        {
            assert!(m >= t);
        }
    }
}
