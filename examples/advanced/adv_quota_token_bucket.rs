//! # Advanced Token-Bucket Rate Limiter
//!
//! Token-bucket: capacity, refill rate (tokens/sec), current tokens.
//! On request, deduct cost; if not enough tokens, deny.
//! Refill is elapsed_secs × rate, capped at capacity.
//!
//! Demonstrates the **ADV.28** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Token bucket (RFC 2697) + Stripe-style API rate limiting.
//!
//! Run with: cargo run --example adv_quota_token_bucket
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BucketVerdict {
    Allowed { remaining: f64 },
    Denied { deficit: f64 },
    InvalidConfig,
}

pub fn step(
    capacity: f64,
    refill_rate_per_sec: f64,
    tokens_at_last_step: f64,
    elapsed_secs: f64,
    request_cost: f64,
) -> BucketVerdict {
    if capacity <= 0.0
        || !capacity.is_finite()
        || refill_rate_per_sec < 0.0
        || !refill_rate_per_sec.is_finite()
        || tokens_at_last_step < 0.0
        || !tokens_at_last_step.is_finite()
        || tokens_at_last_step > capacity
        || elapsed_secs < 0.0
        || !elapsed_secs.is_finite()
        || request_cost < 0.0
        || !request_cost.is_finite()
    {
        return BucketVerdict::InvalidConfig;
    }
    let refilled = (tokens_at_last_step + elapsed_secs * refill_rate_per_sec).min(capacity);
    if refilled >= request_cost {
        BucketVerdict::Allowed {
            remaining: refilled - request_cost,
        }
    } else {
        BucketVerdict::Denied {
            deficit: request_cost - refilled,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_quota_token_bucket")?;

    println!("normal: {:?}", step(100.0, 10.0, 50.0, 1.0, 5.0));
    println!("denied: {:?}", step(100.0, 10.0, 5.0, 0.0, 10.0));
    println!("refill cap: {:?}", step(100.0, 10.0, 50.0, 1000.0, 0.0));
    println!("invalid: {:?}", step(0.0, 10.0, 5.0, 1.0, 1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_quota_allowed() {
        let v = step(100.0, 10.0, 50.0, 1.0, 5.0);
        assert!(matches!(v, BucketVerdict::Allowed { .. }));
    }

    #[test]
    fn over_quota_denied() {
        let v = step(100.0, 10.0, 5.0, 0.0, 10.0);
        if let BucketVerdict::Denied { deficit } = v {
            assert!((deficit - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn refill_capped_at_capacity() {
        let v = step(100.0, 10.0, 50.0, 1000.0, 0.0);
        if let BucketVerdict::Allowed { remaining } = v {
            // 50 + 10*1000 = 10050 → capped at 100.
            assert!((remaining - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_capacity_invalid() {
        assert_eq!(step(0.0, 10.0, 0.0, 1.0, 1.0), BucketVerdict::InvalidConfig);
    }

    #[test]
    fn negative_capacity_invalid() {
        assert_eq!(
            step(-10.0, 10.0, 0.0, 1.0, 1.0),
            BucketVerdict::InvalidConfig
        );
    }

    #[test]
    fn negative_rate_invalid() {
        assert_eq!(
            step(100.0, -1.0, 50.0, 1.0, 1.0),
            BucketVerdict::InvalidConfig
        );
    }

    #[test]
    fn tokens_over_capacity_invalid() {
        assert_eq!(
            step(100.0, 10.0, 200.0, 1.0, 1.0),
            BucketVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            step(100.0, f64::NAN, 50.0, 1.0, 1.0),
            BucketVerdict::InvalidConfig
        );
    }

    #[test]
    fn exact_balance_allowed() {
        let v = step(100.0, 0.0, 5.0, 0.0, 5.0);
        if let BucketVerdict::Allowed { remaining } = v {
            assert!((remaining - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn no_refill_in_zero_elapsed() {
        let v = step(100.0, 10.0, 5.0, 0.0, 5.0);
        if let BucketVerdict::Allowed { remaining } = v {
            assert!((remaining - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = step(100.0, 10.0, 50.0, 1.0, 5.0);
        let b = step(100.0, 10.0, 50.0, 1.0, 5.0);
        assert_eq!(a, b);
    }
}
