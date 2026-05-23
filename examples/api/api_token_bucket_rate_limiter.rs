//! # API Token-Bucket Rate Limiter
//!
//! Token-bucket smooths bursty request loads: capacity B, refill rate
//! R tokens/sec. Each request consumes tokens equal to its cost; when
//! bucket empty, request is rejected. Constraints: B ≥ 1, R > 0. This
//! recipe builds the take/refill state machine.
//!
//! Demonstrates the **API.5** recipe for PMAT-125 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tanenbaum & Wetherall (2011). Computer Networks (5th ed.) §5.4.2.
//!
//! Run with: cargo run --example api_token_bucket_rate_limiter
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct Bucket {
    pub capacity: u32,
    pub refill_per_sec: f64,
    pub current_tokens: f64,
    pub last_refill_ms: u64,
}

#[derive(Debug, PartialEq)]
pub enum TakeVerdict {
    Allowed { remaining: f64 },
    Throttled { wait_ms: u64 },
    InvalidConfig,
    InvalidCost,
}

pub fn refill(bucket: &mut Bucket, now_ms: u64) {
    if now_ms <= bucket.last_refill_ms {
        return;
    }
    let elapsed_secs = (now_ms - bucket.last_refill_ms) as f64 / 1000.0;
    let added = elapsed_secs * bucket.refill_per_sec;
    bucket.current_tokens = (bucket.current_tokens + added).min(f64::from(bucket.capacity));
    bucket.last_refill_ms = now_ms;
}

pub fn take(bucket: &mut Bucket, cost: u32, now_ms: u64) -> TakeVerdict {
    if bucket.capacity == 0 || bucket.refill_per_sec <= 0.0 || !bucket.refill_per_sec.is_finite() {
        return TakeVerdict::InvalidConfig;
    }
    if cost == 0 || cost > bucket.capacity {
        return TakeVerdict::InvalidCost;
    }
    refill(bucket, now_ms);
    let cost_f = f64::from(cost);
    if bucket.current_tokens >= cost_f {
        bucket.current_tokens -= cost_f;
        TakeVerdict::Allowed {
            remaining: bucket.current_tokens,
        }
    } else {
        let deficit = cost_f - bucket.current_tokens;
        let wait_secs = deficit / bucket.refill_per_sec;
        let wait_ms = (wait_secs * 1000.0).ceil() as u64;
        TakeVerdict::Throttled { wait_ms }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_token_bucket_rate_limiter")?;

    let mut bucket = Bucket {
        capacity: 10,
        refill_per_sec: 2.0,
        current_tokens: 10.0,
        last_refill_ms: 0,
    };
    for cost in [1, 5, 5, 1] {
        println!("take {cost}: {:?}", take(&mut bucket, cost, 0));
    }
    println!("after 1s: {:?}", take(&mut bucket, 3, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_bucket() -> Bucket {
        Bucket {
            capacity: 10,
            refill_per_sec: 2.0,
            current_tokens: 10.0,
            last_refill_ms: 0,
        }
    }

    #[test]
    fn limiter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_bucket_allows_request() {
        let mut b = full_bucket();
        let v = take(&mut b, 5, 0);
        assert!(matches!(v, TakeVerdict::Allowed { .. }));
    }

    #[test]
    fn empty_bucket_throttles() {
        let mut b = full_bucket();
        b.current_tokens = 0.0;
        let v = take(&mut b, 5, 0);
        assert!(matches!(v, TakeVerdict::Throttled { .. }));
    }

    #[test]
    fn refill_after_elapsed_time() {
        let mut b = full_bucket();
        b.current_tokens = 0.0;
        // 1s × 2 tokens/sec = 2 tokens added.
        refill(&mut b, 1000);
        assert!((b.current_tokens - 2.0).abs() < 1e-9);
    }

    #[test]
    fn refill_caps_at_capacity() {
        let mut b = full_bucket();
        b.current_tokens = 0.0;
        // 100s × 2 tokens/sec = 200 added, but capacity is 10.
        refill(&mut b, 100_000);
        assert_eq!(b.current_tokens, 10.0);
    }

    #[test]
    fn cost_exceeds_capacity_invalid() {
        let mut b = full_bucket();
        let v = take(&mut b, 100, 0);
        assert_eq!(v, TakeVerdict::InvalidCost);
    }

    #[test]
    fn zero_cost_invalid() {
        let mut b = full_bucket();
        assert_eq!(take(&mut b, 0, 0), TakeVerdict::InvalidCost);
    }

    #[test]
    fn invalid_config_rejected() {
        let mut b = full_bucket();
        b.capacity = 0;
        assert_eq!(take(&mut b, 1, 0), TakeVerdict::InvalidConfig);
        b = full_bucket();
        b.refill_per_sec = 0.0;
        assert_eq!(take(&mut b, 1, 0), TakeVerdict::InvalidConfig);
    }

    #[test]
    fn wait_ms_proportional_to_deficit() {
        let mut b = full_bucket();
        b.current_tokens = 1.0;
        // Need 5, have 1, deficit 4, refill 2/sec → wait 2 sec = 2000 ms.
        let v = take(&mut b, 5, 0);
        assert_eq!(v, TakeVerdict::Throttled { wait_ms: 2000 });
    }

    #[test]
    fn sequential_takes_drain_bucket() {
        let mut b = full_bucket();
        let _ = take(&mut b, 4, 0);
        let _ = take(&mut b, 4, 0);
        // 10 - 4 - 4 = 2 left.
        if let TakeVerdict::Allowed { remaining } = take(&mut b, 1, 0) {
            assert!((remaining - 1.0).abs() < 1e-9);
        }
    }
}
