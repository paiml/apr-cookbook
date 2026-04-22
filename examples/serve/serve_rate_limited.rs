//! # Recipe: HTTP Model Serve with Rate-Limit + Auth
//!
//! **Category**: serve
//! **CLI Equivalent**: `apr serve --rate-limit 10/s --auth bearer`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example serve_rate_limited` exits 0
//! 2. [x] `cargo test --example serve_rate_limited` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr serve` request pipeline in-process (no sockets)
//! 10. [x] Unit tests cover token bucket, auth accept/reject, burst limit
//!
//! ## Learning Objective
//! Demonstrates the request-handling middleware chain used by `apr serve`:
//! bearer-token authentication then a simple token-bucket rate limiter. We
//! simulate a burst of 20 requests from a single client and record which are
//! accepted, rate-limited, or rejected for bad credentials.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serve_rate_limited
//! ```
//!
//! ## References
//! - Fielding, R. et al. (1999). *RFC 2616: Hypertext Transfer Protocol — HTTP/1.1*. IETF. URL: https://www.rfc-editor.org/rfc/rfc2616
//! - Adya, A. et al. (2019). *Centrifuge: A Reconfigurable Distributed Token Bucket*. SOSP. arXiv:1911.07028

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Unauthorized,
    RateLimited,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub token: Option<String>,
    pub time_ms: u64,
}

pub struct TokenBucket {
    pub capacity: u32,
    pub refill_per_sec: u32,
    last_refill_ms: u64,
    tokens: f64,
}

impl TokenBucket {
    pub fn new(capacity: u32, refill_per_sec: u32) -> Self {
        Self {
            capacity,
            refill_per_sec,
            last_refill_ms: 0,
            tokens: f64::from(capacity),
        }
    }

    pub fn try_acquire(&mut self, now_ms: u64) -> bool {
        let elapsed_sec = (now_ms.saturating_sub(self.last_refill_ms)) as f64 / 1000.0;
        self.tokens = (self.tokens + elapsed_sec * f64::from(self.refill_per_sec))
            .min(f64::from(self.capacity));
        self.last_refill_ms = now_ms;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

pub fn validate_bearer(token: Option<&str>, allowed: &[&str]) -> bool {
    match token {
        Some(t) => allowed.contains(&t),
        None => false,
    }
}

pub fn handle_request(req: &Request, bucket: &mut TokenBucket, allowed_tokens: &[&str]) -> Outcome {
    if !validate_bearer(req.token.as_deref(), allowed_tokens) {
        return Outcome::Unauthorized;
    }
    if !bucket.try_acquire(req.time_ms) {
        return Outcome::RateLimited;
    }
    Outcome::Ok
}

fn simulate_burst() -> Vec<Request> {
    (0..20)
        .map(|i| Request {
            id: i,
            token: if i == 7 {
                Some("bad-token".into())
            } else {
                Some("secret-bearer-abc".into())
            },
            // First 10 at t=0, next 10 spaced 50ms apart.
            time_ms: if i < 10 { 0 } else { 50 * (i - 10) },
        })
        .collect()
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("serve_rate_limited")?;
    println!("=== Recipe: {} ===", ctx.name());

    let allowed = ["secret-bearer-abc"];
    let mut bucket = TokenBucket::new(10, 10);
    let reqs = simulate_burst();

    let mut counts = [0u32; 3];
    let mut transcript = Vec::new();
    for r in &reqs {
        let outcome = handle_request(r, &mut bucket, &allowed);
        match outcome {
            Outcome::Ok => counts[0] += 1,
            Outcome::Unauthorized => counts[1] += 1,
            Outcome::RateLimited => counts[2] += 1,
        }
        transcript.push(json!({
            "id": r.id,
            "time_ms": r.time_ms,
            "outcome": match outcome {
                Outcome::Ok => "ok",
                Outcome::Unauthorized => "unauthorized",
                Outcome::RateLimited => "rate_limited",
            }
        }));
    }

    println!("Requests processed: {}", reqs.len());
    println!("  OK:           {}", counts[0]);
    println!("  Unauthorized: {}", counts[1]);
    println!("  Rate-limited: {}", counts[2]);

    let report = json!({
        "recipe": ctx.name(),
        "rate_limit": "10/s (burst 10)",
        "auth": "bearer",
        "n_total": reqs.len(),
        "n_ok": counts[0],
        "n_unauthorized": counts[1],
        "n_rate_limited": counts[2],
        "transcript": transcript,
    });
    let out = ctx.path("serve-rate-limit.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_accepts_up_to_capacity() {
        let mut b = TokenBucket::new(3, 0);
        assert!(b.try_acquire(0));
        assert!(b.try_acquire(0));
        assert!(b.try_acquire(0));
        assert!(!b.try_acquire(0));
    }

    #[test]
    fn bucket_refills_over_time() {
        let mut b = TokenBucket::new(1, 10);
        assert!(b.try_acquire(0));
        assert!(!b.try_acquire(0));
        // After 150ms at 10/s we should have > 1 token.
        assert!(b.try_acquire(150));
    }

    #[test]
    fn validate_bearer_accepts_known() {
        assert!(validate_bearer(Some("ok"), &["ok", "also-ok"]));
    }

    #[test]
    fn validate_bearer_rejects_unknown() {
        assert!(!validate_bearer(Some("nope"), &["ok"]));
        assert!(!validate_bearer(None, &["ok"]));
    }

    #[test]
    fn handle_rejects_bad_token() {
        let mut b = TokenBucket::new(10, 10);
        let req = Request {
            id: 1,
            token: Some("wrong".into()),
            time_ms: 0,
        };
        assert_eq!(
            handle_request(&req, &mut b, &["right"]),
            Outcome::Unauthorized
        );
    }
}
