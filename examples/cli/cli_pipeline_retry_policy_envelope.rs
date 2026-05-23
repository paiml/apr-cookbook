//! # apr pipeline --retry — Retry Policy Envelope
//!
//! Pipeline stages can retry transient failures. Policy: max_attempts
//! ∈ [1, 10] (≥ 11 risks runaway), initial_backoff ∈ [100ms, 60s],
//! exponential_base ∈ [1.5, 4.0] (1.0 = constant, > 4 too aggressive).
//! This recipe builds the envelope + delay calculator.
//!
//! Demonstrates the **PIPE.4** recipe for PMAT-121 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPE-001 + AWS retry guidance (exponential backoff + jitter)
//!
//! Run with: cargo run --example cli_pipeline_retry_policy_envelope
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PolicyVerdict {
    Ok,
    AttemptsBelowMin,
    AttemptsAboveMax,
    BackoffOutOfRange,
    BaseOutOfRange,
}

const MAX_ATTEMPTS: u32 = 10;
const MIN_BACKOFF_MS: u32 = 100;
const MAX_BACKOFF_MS: u32 = 60_000;

pub fn validate(max_attempts: u32, initial_backoff_ms: u32, exp_base: f64) -> PolicyVerdict {
    if max_attempts == 0 {
        return PolicyVerdict::AttemptsBelowMin;
    }
    if max_attempts > MAX_ATTEMPTS {
        return PolicyVerdict::AttemptsAboveMax;
    }
    if !(MIN_BACKOFF_MS..=MAX_BACKOFF_MS).contains(&initial_backoff_ms) {
        return PolicyVerdict::BackoffOutOfRange;
    }
    if !exp_base.is_finite() || !(1.5..=4.0).contains(&exp_base) {
        return PolicyVerdict::BaseOutOfRange;
    }
    PolicyVerdict::Ok
}

pub fn delay_ms_for_attempt(attempt: u32, initial_backoff_ms: u32, exp_base: f64) -> Option<u64> {
    if attempt == 0 || !exp_base.is_finite() || exp_base <= 0.0 {
        return None;
    }
    let multiplier = exp_base.powi((attempt - 1) as i32);
    let delay = f64::from(initial_backoff_ms) * multiplier;
    if !delay.is_finite() {
        return None;
    }
    Some(delay.min(u64::MAX as f64) as u64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_retry_policy_envelope")?;

    let cases = [
        (3u32, 500u32, 2.0),
        (0, 500, 2.0),
        (15, 500, 2.0),
        (3, 50, 2.0),
        (3, 500, 1.0),
        (3, 500, 5.0),
    ];
    for (a, b, base) in cases {
        println!(
            "attempts={a} backoff={b}ms base={base}  →  {:?}",
            validate(a, b, base)
        );
    }
    for n in 1..=5 {
        println!("delay #{n}: {:?}ms", delay_ms_for_attempt(n, 500, 2.0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_policy_passes() {
        assert_eq!(validate(3, 500, 2.0), PolicyVerdict::Ok);
    }

    #[test]
    fn zero_attempts_rejected() {
        assert_eq!(validate(0, 500, 2.0), PolicyVerdict::AttemptsBelowMin);
    }

    #[test]
    fn over_max_attempts_rejected() {
        assert_eq!(validate(15, 500, 2.0), PolicyVerdict::AttemptsAboveMax);
    }

    #[test]
    fn at_min_backoff_passes() {
        assert_eq!(validate(3, MIN_BACKOFF_MS, 2.0), PolicyVerdict::Ok);
    }

    #[test]
    fn under_min_backoff_rejected() {
        assert_eq!(validate(3, 50, 2.0), PolicyVerdict::BackoffOutOfRange);
    }

    #[test]
    fn over_max_backoff_rejected() {
        assert_eq!(validate(3, 100_000, 2.0), PolicyVerdict::BackoffOutOfRange);
    }

    #[test]
    fn base_below_range_rejected() {
        // base=1.0 → constant, no exponential growth.
        assert_eq!(validate(3, 500, 1.0), PolicyVerdict::BaseOutOfRange);
    }

    #[test]
    fn base_above_range_rejected() {
        assert_eq!(validate(3, 500, 5.0), PolicyVerdict::BaseOutOfRange);
    }

    #[test]
    fn delay_grows_exponentially() {
        let d1 = delay_ms_for_attempt(1, 500, 2.0).unwrap();
        let d2 = delay_ms_for_attempt(2, 500, 2.0).unwrap();
        let d3 = delay_ms_for_attempt(3, 500, 2.0).unwrap();
        assert_eq!(d1, 500);
        assert_eq!(d2, 1000);
        assert_eq!(d3, 2000);
    }

    #[test]
    fn delay_zero_attempt_yields_none() {
        assert!(delay_ms_for_attempt(0, 500, 2.0).is_none());
    }
}
