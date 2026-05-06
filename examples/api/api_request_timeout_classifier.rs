//! # API Request Timeout Classifier
//!
//! Per-route timeout policy:
//!   /healthz → 1s (Fast)
//!   /predict (cached) → 5s (Medium)
//!   /predict (cold model) → 30s (Slow)
//!   /batch → 5min (Extreme)
//!
//! This recipe maps (route, cache_state) to a tier + timeout.
//!
//! Demonstrates the **API.10** recipe for PMAT-138 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Envoy proxy per-route timeout best-practice.
//!
//! Run with: cargo run --example api_request_timeout_classifier
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyTier {
    Fast,
    Medium,
    Slow,
    Extreme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    Hot,
    Cold,
    NotApplicable,
}

#[derive(Debug, PartialEq)]
pub enum TimeoutVerdict {
    Ok {
        tier: LatencyTier,
        timeout_secs: u32,
    },
    UnknownRoute,
    EmptyRoute,
}

pub fn classify(route: &str, cache: CacheState) -> TimeoutVerdict {
    if route.is_empty() {
        return TimeoutVerdict::EmptyRoute;
    }
    let (tier, timeout_secs) = match (route, cache) {
        ("/healthz" | "/livez" | "/readyz", _) => (LatencyTier::Fast, 1),
        ("/predict", CacheState::Hot) => (LatencyTier::Medium, 5),
        ("/predict", CacheState::Cold) => (LatencyTier::Slow, 30),
        ("/batch", _) => (LatencyTier::Extreme, 300),
        _ => return TimeoutVerdict::UnknownRoute,
    };
    TimeoutVerdict::Ok { tier, timeout_secs }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_request_timeout_classifier")?;

    let cases = [
        ("/healthz", CacheState::NotApplicable),
        ("/predict", CacheState::Hot),
        ("/predict", CacheState::Cold),
        ("/batch", CacheState::NotApplicable),
        ("/unknown", CacheState::Hot),
    ];
    for (r, c) in cases {
        println!("{r} {c:?} → {:?}", classify(r, c));
    }
    println!("empty: {:?}", classify("", CacheState::Hot));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthz_is_fast() {
        let v = classify("/healthz", CacheState::NotApplicable);
        assert_eq!(
            v,
            TimeoutVerdict::Ok {
                tier: LatencyTier::Fast,
                timeout_secs: 1
            }
        );
    }

    #[test]
    fn livez_readyz_also_fast() {
        assert!(matches!(
            classify("/livez", CacheState::NotApplicable),
            TimeoutVerdict::Ok {
                tier: LatencyTier::Fast,
                ..
            }
        ));
        assert!(matches!(
            classify("/readyz", CacheState::NotApplicable),
            TimeoutVerdict::Ok {
                tier: LatencyTier::Fast,
                ..
            }
        ));
    }

    #[test]
    fn predict_hot_medium() {
        let v = classify("/predict", CacheState::Hot);
        assert_eq!(
            v,
            TimeoutVerdict::Ok {
                tier: LatencyTier::Medium,
                timeout_secs: 5
            }
        );
    }

    #[test]
    fn predict_cold_slow() {
        let v = classify("/predict", CacheState::Cold);
        assert_eq!(
            v,
            TimeoutVerdict::Ok {
                tier: LatencyTier::Slow,
                timeout_secs: 30
            }
        );
    }

    #[test]
    fn batch_extreme() {
        let v = classify("/batch", CacheState::NotApplicable);
        assert_eq!(
            v,
            TimeoutVerdict::Ok {
                tier: LatencyTier::Extreme,
                timeout_secs: 300
            }
        );
    }

    #[test]
    fn unknown_route_rejected() {
        let v = classify("/random", CacheState::Hot);
        assert_eq!(v, TimeoutVerdict::UnknownRoute);
    }

    #[test]
    fn empty_route_rejected() {
        assert_eq!(classify("", CacheState::Hot), TimeoutVerdict::EmptyRoute);
    }

    #[test]
    fn cold_takes_six_times_hot() {
        let hot = classify("/predict", CacheState::Hot);
        let cold = classify("/predict", CacheState::Cold);
        if let (
            TimeoutVerdict::Ok {
                timeout_secs: t_hot,
                ..
            },
            TimeoutVerdict::Ok {
                timeout_secs: t_cold,
                ..
            },
        ) = (hot, cold)
        {
            assert_eq!(t_cold, t_hot * 6);
        }
    }

    #[test]
    fn case_sensitive_path() {
        // /HEALTHZ is not /healthz.
        let v = classify("/HEALTHZ", CacheState::NotApplicable);
        assert_eq!(v, TimeoutVerdict::UnknownRoute);
    }

    #[test]
    fn batch_ignores_cache_state() {
        // /batch always Extreme regardless of cache.
        let v_hot = classify("/batch", CacheState::Hot);
        let v_cold = classify("/batch", CacheState::Cold);
        assert_eq!(v_hot, v_cold);
    }
}
