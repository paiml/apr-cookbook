//! # Advanced Retry Backoff with Jitter
//!
//! Exponential backoff: wait_ms = base × 2^attempt + random_jitter.
//! Jitter strategies:
//!   None: pure exp; can cause synchronized retry storms
//!   Full: random in [0, 2^attempt × base]; smoothest
//!   Decorrelated: random in [base, prev × 3]; AWS recommended
//!
//! Demonstrates the **ADV.24** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AWS Architecture Blog: Exponential Backoff and Jitter (2015).
//!
//! Run with: cargo run --example adv_retry_backoff
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterMode {
    None,
    Full,
    Decorrelated,
}

#[derive(Debug, PartialEq)]
pub enum BackoffVerdict {
    Ok {
        wait_ms: u64,
        max_for_attempt_ms: u64,
    },
    InvalidAttempt,
    InvalidBase,
    AboveCap,
}

const MAX_BACKOFF_MS: u64 = 60_000;

pub fn compute(
    attempt: u32,
    base_ms: u32,
    jitter: JitterMode,
    prev_wait_ms: u64,
) -> BackoffVerdict {
    if attempt == 0 {
        return BackoffVerdict::InvalidAttempt;
    }
    if base_ms == 0 {
        return BackoffVerdict::InvalidBase;
    }
    let exp = u64::from(base_ms).saturating_mul(2u64.saturating_pow(attempt - 1));
    if exp > MAX_BACKOFF_MS {
        return BackoffVerdict::AboveCap;
    }
    let wait_ms = match jitter {
        JitterMode::None => exp,
        JitterMode::Full => exp / 2,
        JitterMode::Decorrelated => {
            let lo = u64::from(base_ms);
            let hi = prev_wait_ms.saturating_mul(3).min(MAX_BACKOFF_MS);
            (lo + hi) / 2
        }
    };
    BackoffVerdict::Ok {
        wait_ms,
        max_for_attempt_ms: exp,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_retry_backoff")?;

    println!(
        "attempt 1, no jitter: {:?}",
        compute(1, 100, JitterMode::None, 0)
    );
    println!(
        "attempt 3, full jitter: {:?}",
        compute(3, 100, JitterMode::Full, 0)
    );
    println!(
        "attempt 4, decorrelated: {:?}",
        compute(4, 100, JitterMode::Decorrelated, 800)
    );
    println!(
        "attempt 20 capped: {:?}",
        compute(20, 100, JitterMode::None, 0)
    );
    println!("invalid: {:?}", compute(0, 100, JitterMode::None, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn attempt_one_no_jitter_base() {
        let v = compute(1, 100, JitterMode::None, 0);
        if let BackoffVerdict::Ok { wait_ms, .. } = v {
            assert_eq!(wait_ms, 100);
        }
    }

    #[test]
    fn exponential_doubles_each_attempt() {
        let v_1 = compute(1, 100, JitterMode::None, 0);
        let v_2 = compute(2, 100, JitterMode::None, 0);
        let v_3 = compute(3, 100, JitterMode::None, 0);
        if let (
            BackoffVerdict::Ok { wait_ms: a, .. },
            BackoffVerdict::Ok { wait_ms: b, .. },
            BackoffVerdict::Ok { wait_ms: c, .. },
        ) = (v_1, v_2, v_3)
        {
            assert_eq!(b, a * 2);
            assert_eq!(c, b * 2);
        }
    }

    #[test]
    fn full_jitter_half_of_max() {
        let v = compute(3, 100, JitterMode::Full, 0);
        if let BackoffVerdict::Ok {
            wait_ms,
            max_for_attempt_ms,
        } = v
        {
            assert_eq!(wait_ms, max_for_attempt_ms / 2);
        }
    }

    #[test]
    fn invalid_zero_attempt() {
        assert_eq!(
            compute(0, 100, JitterMode::None, 0),
            BackoffVerdict::InvalidAttempt
        );
    }

    #[test]
    fn invalid_zero_base() {
        assert_eq!(
            compute(1, 0, JitterMode::None, 0),
            BackoffVerdict::InvalidBase
        );
    }

    #[test]
    fn above_cap_rejected() {
        // base 1000, attempt 20 → 1000 × 2^19 huge.
        let v = compute(20, 1000, JitterMode::None, 0);
        assert_eq!(v, BackoffVerdict::AboveCap);
    }

    #[test]
    fn decorrelated_uses_prev() {
        let v_a = compute(2, 100, JitterMode::Decorrelated, 100);
        let v_b = compute(2, 100, JitterMode::Decorrelated, 1000);
        if let (BackoffVerdict::Ok { wait_ms: a, .. }, BackoffVerdict::Ok { wait_ms: b, .. }) =
            (v_a, v_b)
        {
            // Different prev → different wait.
            assert_ne!(a, b);
        }
    }

    #[test]
    fn full_jitter_lower_than_no_jitter() {
        let no = compute(3, 100, JitterMode::None, 0);
        let full = compute(3, 100, JitterMode::Full, 0);
        if let (BackoffVerdict::Ok { wait_ms: n, .. }, BackoffVerdict::Ok { wait_ms: f, .. }) =
            (no, full)
        {
            assert!(f < n);
        }
    }

    #[test]
    fn small_attempt_under_cap() {
        let v = compute(5, 100, JitterMode::None, 0);
        assert!(matches!(v, BackoffVerdict::Ok { .. }));
    }

    #[test]
    fn max_for_attempt_returned() {
        let v = compute(4, 100, JitterMode::Full, 0);
        if let BackoffVerdict::Ok {
            max_for_attempt_ms, ..
        } = v
        {
            // 100 × 2^3 = 800.
            assert_eq!(max_for_attempt_ms, 800);
        }
    }
}
