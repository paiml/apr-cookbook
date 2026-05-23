//! # Advanced Warm-Up Phase Classifier
//!
//! After model load, latency is dominated by JIT compilation, kernel
//! init, page faults. Classify whether the inference server has exited
//! warm-up:
//!   <100 requests OR p50 dropping → still warming
//!   ≥100 requests AND p50 stable → ready
//!
//! Demonstrates the **ADV.30** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TensorRT model warm-up best practices.
//!
//! Run with: cargo run --example adv_warmup_classifier
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ready { stable_p50_ms: f64 },
    StillWarmingTooFew { request_count: u32 },
    StillWarmingP50Dropping { drop_pct: f64 },
    InvalidLatencies,
}

pub fn classify(request_count: u32, recent_p50_ms: f64, older_p50_ms: f64) -> WarmupVerdict {
    if !recent_p50_ms.is_finite()
        || !older_p50_ms.is_finite()
        || recent_p50_ms < 0.0
        || older_p50_ms <= 0.0
    {
        return WarmupVerdict::InvalidLatencies;
    }
    if request_count < 100 {
        return WarmupVerdict::StillWarmingTooFew { request_count };
    }
    let drop_pct = (older_p50_ms - recent_p50_ms) / older_p50_ms;
    if drop_pct > 0.10 {
        return WarmupVerdict::StillWarmingP50Dropping { drop_pct };
    }
    WarmupVerdict::Ready {
        stable_p50_ms: recent_p50_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_warmup_classifier")?;

    println!("too few: {:?}", classify(50, 100.0, 200.0));
    println!("still dropping: {:?}", classify(200, 100.0, 200.0));
    println!("ready: {:?}", classify(200, 95.0, 100.0));
    println!("invalid: {:?}", classify(100, -1.0, 100.0));
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
    fn few_requests_warming() {
        let v = classify(50, 100.0, 200.0);
        assert!(matches!(v, WarmupVerdict::StillWarmingTooFew { .. }));
    }

    #[test]
    fn p50_dropping_warming() {
        // 200 → 100 = 50% drop.
        let v = classify(200, 100.0, 200.0);
        assert!(matches!(v, WarmupVerdict::StillWarmingP50Dropping { .. }));
    }

    #[test]
    fn stable_p50_ready() {
        // 100 → 95 = 5% drop, below 10% threshold.
        let v = classify(200, 95.0, 100.0);
        assert!(matches!(v, WarmupVerdict::Ready { .. }));
    }

    #[test]
    fn flat_p50_ready() {
        let v = classify(500, 50.0, 50.0);
        assert!(matches!(v, WarmupVerdict::Ready { .. }));
    }

    #[test]
    fn negative_recent_invalid() {
        assert_eq!(classify(100, -1.0, 100.0), WarmupVerdict::InvalidLatencies);
    }

    #[test]
    fn zero_older_invalid() {
        assert_eq!(classify(100, 50.0, 0.0), WarmupVerdict::InvalidLatencies);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            classify(100, f64::NAN, 100.0),
            WarmupVerdict::InvalidLatencies
        );
    }

    #[test]
    fn boundary_at_100_requests() {
        // Exactly 100 requests → not "too few".
        let v = classify(100, 95.0, 100.0);
        assert!(matches!(v, WarmupVerdict::Ready { .. }));
    }

    #[test]
    fn boundary_at_10_pct_drop_ready() {
        // Exactly 10% drop = not "dropping" (only > 10%).
        let v = classify(200, 90.0, 100.0);
        assert!(matches!(v, WarmupVerdict::Ready { .. }));
    }

    #[test]
    fn p50_increasing_still_ready() {
        // P50 went up (negative drop) → stable enough; not warming.
        let v = classify(200, 100.0, 90.0);
        assert!(matches!(v, WarmupVerdict::Ready { .. }));
    }

    #[test]
    fn deterministic() {
        let a = classify(200, 95.0, 100.0);
        let b = classify(200, 95.0, 100.0);
        assert_eq!(a, b);
    }
}
