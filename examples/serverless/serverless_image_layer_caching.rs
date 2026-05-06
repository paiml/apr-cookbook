//! # Serverless Container-Image Layer Caching
//!
//! Lambda container layers cached on warm starts. Cache hit rate
//! depends on:
//! - layer count (more layers = more chances to hit)
//! - layer reuse across functions (shared base layer = high hit)
//! - layer churn (frequent rebuilds = low hit)
//!
//! Picker: predicts hit_rate_pct + tier (Cold/Warm/Hot/Optimal).
//!
//! Demonstrates the **SVL.11** recipe for PMAT-144 (serverless round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Lambda container image layer caching docs.
//!
//! Run with: cargo run --example serverless_image_layer_caching
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Cold,
    Warm,
    Hot,
    Optimal,
}

#[derive(Debug, PartialEq)]
pub enum CacheVerdict {
    Ok {
        predicted_hit_rate_pct: u32,
        tier: CacheTier,
    },
    InvalidLayerCount,
    InvalidReusePct,
    InvalidChurnPct,
}

pub fn predict(layer_count: u32, reuse_pct: u32, churn_pct: u32) -> CacheVerdict {
    if layer_count == 0 || layer_count > 100 {
        return CacheVerdict::InvalidLayerCount;
    }
    if reuse_pct > 100 {
        return CacheVerdict::InvalidReusePct;
    }
    if churn_pct > 100 {
        return CacheVerdict::InvalidChurnPct;
    }
    let raw = reuse_pct.saturating_sub(churn_pct);
    let bonus = (layer_count.min(20) * 2).min(40);
    let predicted_hit_rate_pct = (raw + bonus / 4).min(100);
    let tier = if predicted_hit_rate_pct < 25 {
        CacheTier::Cold
    } else if predicted_hit_rate_pct < 60 {
        CacheTier::Warm
    } else if predicted_hit_rate_pct < 90 {
        CacheTier::Hot
    } else {
        CacheTier::Optimal
    };
    CacheVerdict::Ok {
        predicted_hit_rate_pct,
        tier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serverless_image_layer_caching")?;

    println!("typical 5/80/10: {:?}", predict(5, 80, 10));
    println!("high churn 5/80/70: {:?}", predict(5, 80, 70));
    println!("optimal 20/95/2: {:?}", predict(20, 95, 2));
    println!("cold 1/10/5: {:?}", predict(1, 10, 5));
    println!("invalid: {:?}", predict(0, 50, 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_input_warm_or_hot() {
        let v = predict(5, 80, 10);
        if let CacheVerdict::Ok { tier, .. } = v {
            assert!(matches!(tier, CacheTier::Warm | CacheTier::Hot));
        }
    }

    #[test]
    fn high_churn_demotes_tier() {
        let v_low_churn = predict(5, 80, 5);
        let v_high_churn = predict(5, 80, 70);
        if let (
            CacheVerdict::Ok {
                predicted_hit_rate_pct: low,
                ..
            },
            CacheVerdict::Ok {
                predicted_hit_rate_pct: high,
                ..
            },
        ) = (v_low_churn, v_high_churn)
        {
            assert!(low > high);
        }
    }

    #[test]
    fn optimal_when_all_factors_align() {
        let v = predict(20, 95, 2);
        if let CacheVerdict::Ok { tier, .. } = v {
            assert!(matches!(tier, CacheTier::Optimal | CacheTier::Hot));
        }
    }

    #[test]
    fn cold_when_low_reuse_or_high_churn() {
        let v = predict(1, 10, 5);
        if let CacheVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, CacheTier::Cold);
        }
    }

    #[test]
    fn invalid_zero_layers_rejected() {
        assert_eq!(predict(0, 50, 5), CacheVerdict::InvalidLayerCount);
    }

    #[test]
    fn invalid_excessive_layers_rejected() {
        assert_eq!(predict(101, 50, 5), CacheVerdict::InvalidLayerCount);
    }

    #[test]
    fn invalid_reuse_pct_above_100() {
        assert_eq!(predict(5, 150, 5), CacheVerdict::InvalidReusePct);
    }

    #[test]
    fn invalid_churn_pct_above_100() {
        assert_eq!(predict(5, 50, 150), CacheVerdict::InvalidChurnPct);
    }

    #[test]
    fn hit_rate_capped_at_100() {
        let v = predict(50, 100, 0);
        if let CacheVerdict::Ok {
            predicted_hit_rate_pct,
            ..
        } = v
        {
            assert!(predicted_hit_rate_pct <= 100);
        }
    }

    #[test]
    fn more_layers_higher_bonus() {
        let v_few = predict(2, 50, 10);
        let v_many = predict(20, 50, 10);
        if let (
            CacheVerdict::Ok {
                predicted_hit_rate_pct: f,
                ..
            },
            CacheVerdict::Ok {
                predicted_hit_rate_pct: m,
                ..
            },
        ) = (v_few, v_many)
        {
            assert!(m > f);
        }
    }
}
