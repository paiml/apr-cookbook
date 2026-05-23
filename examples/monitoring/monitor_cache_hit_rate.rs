//! # Monitoring Cache Hit-Rate Calculator
//!
//! Cache effectiveness = hits / (hits + misses). Verdict:
//!   ≥0.90 → Excellent
//!   0.70-0.90 → Good
//!   0.50-0.70 → Marginal
//!   <0.50 → Poor
//!
//! Demonstrates the **MON.41** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Operating Systems concepts (Silberschatz) ch. on caching.
//!
//! Run with: cargo run --example monitor_cache_hit_rate
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CacheVerdict {
    Excellent { rate: f64 },
    Good { rate: f64 },
    Marginal { rate: f64 },
    Poor { rate: f64 },
    NoActivity,
}

pub fn check(hits: u64, misses: u64) -> CacheVerdict {
    let total = hits + misses;
    if total == 0 {
        return CacheVerdict::NoActivity;
    }
    let rate = hits as f64 / total as f64;
    if rate >= 0.90 {
        CacheVerdict::Excellent { rate }
    } else if rate >= 0.70 {
        CacheVerdict::Good { rate }
    } else if rate >= 0.50 {
        CacheVerdict::Marginal { rate }
    } else {
        CacheVerdict::Poor { rate }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_cache_hit_rate")?;

    println!("excellent: {:?}", check(950, 50));
    println!("good: {:?}", check(80, 20));
    println!("marginal: {:?}", check(60, 40));
    println!("poor: {:?}", check(10, 90));
    println!("no activity: {:?}", check(0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn excellent_above_90_pct() {
        let v = check(950, 50);
        assert!(matches!(v, CacheVerdict::Excellent { .. }));
    }

    #[test]
    fn good_70_to_90_pct() {
        let v = check(80, 20);
        assert!(matches!(v, CacheVerdict::Good { .. }));
    }

    #[test]
    fn marginal_50_to_70_pct() {
        let v = check(60, 40);
        assert!(matches!(v, CacheVerdict::Marginal { .. }));
    }

    #[test]
    fn poor_below_50_pct() {
        let v = check(10, 90);
        assert!(matches!(v, CacheVerdict::Poor { .. }));
    }

    #[test]
    fn no_activity_zero_zero() {
        assert_eq!(check(0, 0), CacheVerdict::NoActivity);
    }

    #[test]
    fn boundary_at_90_excellent() {
        // Exactly 90% → excellent.
        let v = check(900, 100);
        assert!(matches!(v, CacheVerdict::Excellent { .. }));
    }

    #[test]
    fn boundary_at_70_good() {
        let v = check(700, 300);
        assert!(matches!(v, CacheVerdict::Good { .. }));
    }

    #[test]
    fn boundary_at_50_marginal() {
        let v = check(500, 500);
        assert!(matches!(v, CacheVerdict::Marginal { .. }));
    }

    #[test]
    fn rate_carries_value() {
        let v = check(950, 50);
        if let CacheVerdict::Excellent { rate } = v {
            assert!((rate - 0.95).abs() < 1e-9);
        }
    }

    #[test]
    fn all_hits_excellent_at_1() {
        let v = check(100, 0);
        if let CacheVerdict::Excellent { rate } = v {
            assert!((rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn all_misses_poor_at_0() {
        let v = check(0, 100);
        if let CacheVerdict::Poor { rate } = v {
            assert!((rate - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(950, 50);
        let b = check(950, 50);
        assert_eq!(a, b);
    }
}
