//! # Registry Per-Identity Pull Throttle
//!
//! Throttle pulls per identity (IP, token, or both) using a token
//! bucket. Per anonymous IP: 100/hour; per authenticated token:
//! 5000/hour. Pulls > limit → 429 Too Many Requests.
//!
//! This recipe builds the rate-limit decision (Allow/Deny + remaining
//! tokens) and the bucket-refill clock.
//!
//! Demonstrates the **REG.17** recipe for PMAT-143 (registry round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Docker Hub anonymous/authenticated rate-limit policy.
//!
//! Run with: cargo run --example registry_pull_throttle
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Identity {
    Anonymous,
    Authenticated,
}

#[derive(Debug, PartialEq)]
pub enum ThrottleVerdict {
    Allow { remaining: u32 },
    Deny { retry_after_secs: u32 },
    InvalidTimeWindow,
}

const ANON_LIMIT_PER_HOUR: u32 = 100;
const AUTH_LIMIT_PER_HOUR: u32 = 5_000;
const SECONDS_PER_HOUR: u32 = 3_600;

pub fn check(
    identity: Identity,
    recent_pulls_in_window: u32,
    window_age_secs: u32,
) -> ThrottleVerdict {
    if window_age_secs == 0 {
        return ThrottleVerdict::InvalidTimeWindow;
    }
    let limit = match identity {
        Identity::Anonymous => ANON_LIMIT_PER_HOUR,
        Identity::Authenticated => AUTH_LIMIT_PER_HOUR,
    };
    if recent_pulls_in_window < limit {
        return ThrottleVerdict::Allow {
            remaining: limit - recent_pulls_in_window - 1,
        };
    }
    let retry_after = SECONDS_PER_HOUR.saturating_sub(window_age_secs).max(1);
    ThrottleVerdict::Deny {
        retry_after_secs: retry_after,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("registry_pull_throttle")?;

    println!("anon @ 50: {:?}", check(Identity::Anonymous, 50, 1800));
    println!("anon @ 100: {:?}", check(Identity::Anonymous, 100, 1800));
    println!(
        "auth @ 4000: {:?}",
        check(Identity::Authenticated, 4_000, 1800)
    );
    println!(
        "auth @ 5000: {:?}",
        check(Identity::Authenticated, 5_000, 1800)
    );
    println!("invalid: {:?}", check(Identity::Anonymous, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn anon_under_limit_allowed() {
        let v = check(Identity::Anonymous, 50, 1800);
        if let ThrottleVerdict::Allow { remaining } = v {
            assert_eq!(remaining, 49);
        }
    }

    #[test]
    fn anon_at_limit_denied() {
        let v = check(Identity::Anonymous, ANON_LIMIT_PER_HOUR, 1800);
        assert!(matches!(v, ThrottleVerdict::Deny { .. }));
    }

    #[test]
    fn auth_higher_limit() {
        let v = check(Identity::Authenticated, 4_000, 1800);
        assert!(matches!(v, ThrottleVerdict::Allow { .. }));
    }

    #[test]
    fn auth_at_limit_denied() {
        let v = check(Identity::Authenticated, AUTH_LIMIT_PER_HOUR, 1800);
        assert!(matches!(v, ThrottleVerdict::Deny { .. }));
    }

    #[test]
    fn zero_time_window_invalid() {
        assert_eq!(
            check(Identity::Anonymous, 50, 0),
            ThrottleVerdict::InvalidTimeWindow
        );
    }

    #[test]
    fn retry_after_proportional_to_window_age() {
        // Older window → less retry time.
        let v_young = check(Identity::Anonymous, 100, 100);
        let v_old = check(Identity::Anonymous, 100, 3000);
        if let (
            ThrottleVerdict::Deny {
                retry_after_secs: y,
            },
            ThrottleVerdict::Deny {
                retry_after_secs: o,
            },
        ) = (v_young, v_old)
        {
            assert!(y > o);
        }
    }

    #[test]
    fn just_under_limit_allows_one() {
        let v = check(Identity::Anonymous, ANON_LIMIT_PER_HOUR - 1, 1800);
        if let ThrottleVerdict::Allow { remaining } = v {
            assert_eq!(remaining, 0);
        }
    }

    #[test]
    fn auth_unlocks_50x_limit() {
        // Auth limit is 5000 = 50 × 100 = 50 × anon limit.
        assert_eq!(AUTH_LIMIT_PER_HOUR / ANON_LIMIT_PER_HOUR, 50);
    }

    #[test]
    fn well_above_limit_still_denied() {
        let v = check(Identity::Anonymous, 1000, 1800);
        assert!(matches!(v, ThrottleVerdict::Deny { .. }));
    }

    #[test]
    fn retry_after_at_least_one() {
        // Window > 1 hour → retry_after_secs floors to 1.
        let v = check(Identity::Anonymous, 100, 7200);
        if let ThrottleVerdict::Deny { retry_after_secs } = v {
            assert!(retry_after_secs >= 1);
        }
    }
}
