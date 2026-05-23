//! # API OAuth Token Refresh Window
//!
//! Tokens have an `expires_at` timestamp. To avoid 401-mid-request,
//! refresh proactively when expiry is within `refresh_window_secs` of
//! now. Returns RefreshNow / NoActionNeeded / AlreadyExpired.
//!
//! Plus tier classifier: Healthy (>5min), Warning (1-5min), Critical
//! (<1min).
//!
//! Demonstrates the **API.14** recipe for PMAT-143 (api round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: OAuth 2.0 Token Refresh best-practice (RFC 6749 §6).
//!
//! Run with: cargo run --example api_oauth_token_refresh_window
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_REFRESH_WINDOW_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthTier {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum TokenVerdict {
    NoActionNeeded {
        tier: HealthTier,
        secs_remaining: u64,
    },
    RefreshNow {
        secs_remaining: u64,
    },
    AlreadyExpired {
        secs_overdue: u64,
    },
    InvalidExpiry,
}

pub fn check(now_secs: u64, expires_at_secs: u64, refresh_window_secs: u64) -> TokenVerdict {
    if expires_at_secs == 0 || refresh_window_secs == 0 {
        return TokenVerdict::InvalidExpiry;
    }
    if now_secs >= expires_at_secs {
        return TokenVerdict::AlreadyExpired {
            secs_overdue: now_secs - expires_at_secs,
        };
    }
    let remaining = expires_at_secs - now_secs;
    if remaining <= refresh_window_secs {
        return TokenVerdict::RefreshNow {
            secs_remaining: remaining,
        };
    }
    let tier = if remaining > 300 {
        HealthTier::Healthy
    } else if remaining > 60 {
        HealthTier::Warning
    } else {
        HealthTier::Critical
    };
    TokenVerdict::NoActionNeeded {
        tier,
        secs_remaining: remaining,
    }
}

pub fn check_default(now_secs: u64, expires_at_secs: u64) -> TokenVerdict {
    check(now_secs, expires_at_secs, DEFAULT_REFRESH_WINDOW_SECS)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_oauth_token_refresh_window")?;

    println!("fresh (1h left): {:?}", check_default(1000, 1000 + 3600));
    println!("near expiry (60s): {:?}", check_default(1000, 1060));
    println!("expired: {:?}", check_default(1000, 500));
    println!("invalid: {:?}", check_default(0, 0));
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
    fn fresh_token_no_action() {
        let v = check_default(1000, 1000 + 3600);
        assert!(matches!(
            v,
            TokenVerdict::NoActionNeeded {
                tier: HealthTier::Healthy,
                ..
            }
        ));
    }

    #[test]
    fn near_expiry_refresh_now() {
        let v = check_default(1000, 1060);
        assert!(matches!(v, TokenVerdict::RefreshNow { .. }));
    }

    #[test]
    fn expired_returns_expired() {
        let v = check_default(1000, 500);
        if let TokenVerdict::AlreadyExpired { secs_overdue } = v {
            assert_eq!(secs_overdue, 500);
        }
    }

    #[test]
    fn at_expiry_treated_as_expired() {
        // now == expires_at: AlreadyExpired (avoid race).
        assert!(matches!(
            check_default(1000, 1000),
            TokenVerdict::AlreadyExpired { .. }
        ));
    }

    #[test]
    fn at_refresh_window_triggers_refresh() {
        // Exactly 300s remaining → RefreshNow.
        let v = check_default(1000, 1300);
        assert!(matches!(v, TokenVerdict::RefreshNow { .. }));
    }

    #[test]
    fn just_above_refresh_window_no_action() {
        let v = check_default(1000, 1301);
        assert!(matches!(v, TokenVerdict::NoActionNeeded { .. }));
    }

    #[test]
    fn warning_tier_61_to_300s() {
        let v = check(1000, 1000 + 200, 60); // 200s remaining > 60s window.
        assert!(matches!(
            v,
            TokenVerdict::NoActionNeeded {
                tier: HealthTier::Warning,
                ..
            }
        ));
    }

    #[test]
    fn critical_tier_below_60s() {
        let v = check(1000, 1000 + 30, 10); // 30s remaining > 10s window.
        assert!(matches!(
            v,
            TokenVerdict::NoActionNeeded {
                tier: HealthTier::Critical,
                ..
            }
        ));
    }

    #[test]
    fn invalid_expires_zero_rejected() {
        assert_eq!(check_default(0, 0), TokenVerdict::InvalidExpiry);
    }

    #[test]
    fn invalid_window_zero_rejected() {
        assert_eq!(check(1000, 2000, 0), TokenVerdict::InvalidExpiry);
    }

    #[test]
    fn custom_short_window() {
        // 30-second refresh window.
        let v = check(1000, 1020, 30);
        assert!(matches!(v, TokenVerdict::RefreshNow { .. }));
    }
}
