//! # Monte-Carlo Token Bucket Refill Rate Match
//!
//! Sim a token bucket with capacity C and refill rate R. Workload
//! requests tokens at burst rate. Verify steady-state output rate
//! converges to R, not the burst rate.
//!
//! Demonstrates the **MC.66** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tanenbaum, Computer Networks §5.4 (token bucket).
//!
//! Run with: cargo run --example mc_token_refill_rate_match
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RefillVerdict {
    Ok {
        served: u32,
        rejected: u32,
        observed_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    seconds: u32,
    capacity: u32,
    refill_per_sec: u32,
    burst_per_sec: u32,
) -> RefillVerdict {
    if seconds == 0 || capacity == 0 || refill_per_sec == 0 {
        return RefillVerdict::InvalidConfig;
    }
    let mut tokens: u32 = capacity;
    let mut served: u32 = 0;
    let mut rejected: u32 = 0;
    for _ in 0..seconds {
        // Refill (capped at capacity).
        tokens = (tokens + refill_per_sec).min(capacity);
        // Serve up to `burst_per_sec` requests.
        for _ in 0..burst_per_sec {
            if tokens > 0 {
                tokens -= 1;
                served += 1;
            } else {
                rejected += 1;
            }
        }
    }
    let observed_rate = f64::from(served) / f64::from(seconds);
    RefillVerdict::Ok {
        served,
        rejected,
        observed_rate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_refill_rate_match")?;

    println!("under capacity: {:?}", simulate(60, 100, 10, 5));
    println!("over capacity: {:?}", simulate(60, 100, 10, 100));
    println!("invalid: {:?}", simulate(0, 100, 10, 50));
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
    fn under_capacity_no_rejections() {
        // burst < refill → all served.
        let v = simulate(60, 100, 10, 5);
        if let RefillVerdict::Ok { rejected, .. } = v {
            assert_eq!(rejected, 0);
        }
    }

    #[test]
    fn over_capacity_steady_state_matches_refill() {
        // burst >> refill → steady-state output ≈ refill.
        let seconds = 1000u32;
        let refill = 10u32;
        let v = simulate(seconds, 100, refill, 10_000);
        if let RefillVerdict::Ok { observed_rate, .. } = v {
            // Allow 10% slack for warmup (initial burst from full bucket).
            let expected = f64::from(refill);
            let bound = expected * 1.1 + 1.0;
            assert!(observed_rate <= bound);
            assert!(observed_rate >= expected);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 100, 10, 50), RefillVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(60, 0, 10, 50), RefillVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_refill() {
        assert_eq!(simulate(60, 100, 0, 50), RefillVerdict::InvalidConfig);
    }

    #[test]
    fn rejected_grows_with_burst() {
        let lo = simulate(60, 100, 10, 5);
        let hi = simulate(60, 100, 10, 100);
        if let (RefillVerdict::Ok { rejected: l, .. }, RefillVerdict::Ok { rejected: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn served_le_seconds_times_burst() {
        let v = simulate(60, 100, 10, 50);
        if let RefillVerdict::Ok { served, .. } = v {
            assert!(served <= 60 * 50);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = simulate(60, 100, 10, 50);
        let r2 = simulate(60, 100, 10, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn higher_capacity_more_initial_burst() {
        let small = simulate(10, 5, 5, 100);
        let big = simulate(10, 500, 5, 100);
        if let (RefillVerdict::Ok { served: s, .. }, RefillVerdict::Ok { served: b, .. }) =
            (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn zero_burst_zero_served() {
        let v = simulate(60, 100, 10, 0);
        if let RefillVerdict::Ok {
            served, rejected, ..
        } = v
        {
            assert_eq!(served, 0);
            assert_eq!(rejected, 0);
        }
    }

    #[test]
    fn observed_rate_nonneg() {
        let v = simulate(60, 100, 10, 50);
        if let RefillVerdict::Ok { observed_rate, .. } = v {
            assert!(observed_rate >= 0.0);
        }
    }
}
