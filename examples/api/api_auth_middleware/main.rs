#![allow(unused_imports)]
//! # Recipe: API Authentication and Authorization Middleware
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! API authentication and authorization middleware for model serving:
//! API key validation, token-based auth, request signing (HMAC),
//! role-based access control, rate limiting, and audit logging.
//!
//! ```bash
//! cargo run --example api_auth_middleware
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr serve model.apr          # APR native format
//! apr serve model.gguf         # GGUF (llama.cpp compatible)
//! apr serve model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Recipe: API Auth Middleware ===\n");
    let base_time: u64 = 1_700_000_000;
    let mut mw = AuthMiddleware::new(base_time);

    println!("--- Section 1: Register API Keys ---");
    mw.register_key("key-admin-001", "super-secret-admin", Role::Admin);
    mw.register_key("key-user-042", "user-secret-42", Role::User);
    mw.register_key("key-ro-100", "readonly-secret", Role::ReadOnly);
    for key in mw.keys.values() {
        let perms: Vec<String> = key.permissions.iter().map(ToString::to_string).collect();
        println!(
            "  Registered: {} role={} perms=[{}] rate={}rps",
            key.key_id,
            key.role,
            perms.join(", "),
            key.rate_limit_rps
        );
    }

    println!("\n--- Section 2: Validate API Keys ---");
    for (kid, secret, label) in [
        ("key-admin-001", "super-secret-admin", "valid"),
        ("key-admin-001", "wrong-secret", "bad secret"),
        ("key-unknown", "anything", "unknown"),
    ] {
        println!("  {}: {}", label, mw.validate_key(kid, secret, "10.0.0.1"));
    }

    println!("\n--- Section 3: Token Lifecycle ---");
    let token = mw
        .issue_token("key-user-042", "10.0.0.10")
        .expect("key exists");
    println!(
        "  Issued: {} (expires in {}s)",
        generate_token_string(&token.key_id, base_time),
        TOKEN_LIFETIME_SECS
    );
    println!("  Valid now: {}", mw.validate_token(&token, "10.0.0.10"));
    mw.set_time(base_time + TOKEN_LIFETIME_SECS + 1);
    println!("  After expiry: {}", mw.validate_token(&token, "10.0.0.10"));
    mw.set_time(base_time);

    println!("\n--- Section 4: Request Signing ---");
    let sig = compute_signature("POST", "/v1/models/fraud/infer", base_time);
    let mk_req = |sig: Option<String>, tok: Option<String>| AuthRequest {
        method: "POST".into(),
        path: "/v1/models/fraud/infer".into(),
        key_id: "key-admin-001".into(),
        token: tok,
        signature: sig,
        timestamp: base_time,
    };
    println!(
        "  Signed: {}",
        mw.verify_signature(&mk_req(Some(sig.clone()), None), "10.0.0.1")
    );
    println!(
        "  Tampered: {}",
        mw.verify_signature(&mk_req(Some("0000000000000000".into()), None), "10.0.0.1")
    );
    println!(
        "  Unsigned: {}",
        mw.verify_signature(
            &AuthRequest {
                method: "GET".into(),
                path: "/v1/health".into(),
                key_id: "key-admin-001".into(),
                token: None,
                signature: None,
                timestamp: base_time
            },
            "10.0.0.1"
        )
    );
    println!(
        "  Full auth: {}",
        mw.authenticate_request(
            &mk_req(
                Some(sig),
                Some(generate_token_string("key-admin-001", base_time))
            ),
            "10.0.0.1"
        )
    );
    println!(
        "  Bad token: {}",
        mw.authenticate_request(
            &mk_req(
                Some(compute_signature(
                    "POST",
                    "/v1/models/fraud/infer",
                    base_time
                )),
                Some("bad-format".into())
            ),
            "10.0.0.1"
        )
    );

    println!("\n--- Section 5: Role-Based Access Control ---");
    for (kid, perm, label) in [
        ("key-admin-001", Permission::Admin, "Admin->Admin"),
        ("key-user-042", Permission::Inference, "User->Infer"),
        ("key-user-042", Permission::Admin, "User->Admin"),
        ("key-ro-100", Permission::Inference, "RO->Infer"),
        ("key-ro-100", Permission::Training, "RO->Train"),
    ] {
        println!(
            "  {}: {}",
            label,
            mw.check_permission(kid, perm, "10.0.0.50")
        );
    }

    println!("\n--- Section 6: Rate Limiting & Audit ---");
    let ro_limit = mw
        .keys
        .get("key-ro-100")
        .map_or(DEFAULT_RATE_LIMIT_RPS, |k| k.rate_limit_rps);
    let (mut ra, mut rd) = (0u32, 0u32);
    for _ in 0..(ro_limit + 5) {
        if mw.check_rate_limit("key-ro-100", "10.0.0.100").allowed {
            ra += 1;
        } else {
            rd += 1;
        }
    }
    println!("  Rate limit test: {} allowed, {} denied", ra, rd);
    let (total, allowed, denied) = mw.audit_summary();
    println!(
        "  Audit entries: {} total, {} allowed, {} denied",
        total, allowed, denied
    );
    println!("\n=== Auth middleware demonstration complete ===");
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AuthMiddleware {
        let mut mw = AuthMiddleware::new(1_000_000);
        mw.register_key("admin", "admin-secret", Role::Admin);
        mw.register_key("user", "user-secret", Role::User);
        mw.register_key("readonly", "ro-secret", Role::ReadOnly);
        mw
    }

    #[test]
    fn test_register_keys_and_permissions() {
        let mw = setup();
        assert_eq!(mw.keys.len(), 3);
        let admin = mw.keys.get("admin").unwrap();
        assert!(admin.permissions.contains(&Permission::Admin));
        assert!(admin.permissions.contains(&Permission::Inference));
        let ro = mw.keys.get("readonly").unwrap();
        assert_eq!(ro.permissions, vec![Permission::Inference]);
    }

    #[test]
    fn test_validate_key_scenarios() {
        let mut mw = setup();
        assert!(
            mw.validate_key("admin", "admin-secret", "127.0.0.1")
                .allowed
        );
        assert_eq!(
            mw.validate_key("admin", "wrong", "127.0.0.1").reason,
            "invalid secret"
        );
        assert_eq!(
            mw.validate_key("ghost", "any", "127.0.0.1").reason,
            "unknown key"
        );
    }

    #[test]
    fn test_token_lifecycle() {
        let mut mw = setup();
        assert!(mw.issue_token("ghost", "127.0.0.1").is_none());
        let token = mw.issue_token("user", "127.0.0.1").unwrap();
        assert_eq!(token.key_id, "user");
        assert_eq!(token.expires_at, token.issued_at + TOKEN_LIFETIME_SECS);
        assert!(mw.validate_token(&token, "127.0.0.1").allowed);
        mw.set_time(token.expires_at + 1);
        let result = mw.validate_token(&token, "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "token expired");
    }

    #[test]
    fn test_signature_verification() {
        let mut mw = setup();
        let t = mw.current_time;
        let sig = compute_signature("POST", "/infer", t);
        let mk = |s: Option<String>| AuthRequest {
            method: "POST".into(),
            path: "/infer".into(),
            key_id: "admin".into(),
            token: None,
            signature: s,
            timestamp: t,
        };
        assert!(mw.verify_signature(&mk(Some(sig)), "127.0.0.1").allowed);
        assert_eq!(
            mw.verify_signature(&mk(Some("deadbeef".into())), "127.0.0.1")
                .reason,
            "invalid signature"
        );
        assert_eq!(
            mw.verify_signature(&mk(None), "127.0.0.1").reason,
            "missing signature"
        );
    }

    #[test]
    fn test_rbac() {
        let mut mw = setup();
        assert!(
            mw.check_permission("admin", Permission::ModelManagement, "127.0.0.1")
                .allowed
        );
        assert!(
            !mw.check_permission("user", Permission::Admin, "127.0.0.1")
                .allowed
        );
        assert!(
            !mw.check_permission("readonly", Permission::Training, "127.0.0.1")
                .allowed
        );
    }

    #[test]
    fn test_rate_limit_and_refill() {
        let mut mw = setup();
        assert!(mw.check_rate_limit("user", "127.0.0.1").allowed);
        let limit = mw.keys.get("readonly").unwrap().rate_limit_rps;
        for _ in 0..limit {
            mw.check_rate_limit("readonly", "127.0.0.1");
        }
        assert!(!mw.check_rate_limit("readonly", "127.0.0.1").allowed);
        mw.set_time(mw.current_time + 1);
        assert!(mw.check_rate_limit("readonly", "127.0.0.1").allowed);
    }

    #[test]
    fn test_audit_log() {
        let mut mw = setup();
        mw.validate_key("admin", "admin-secret", "127.0.0.1");
        mw.validate_key("admin", "wrong", "127.0.0.1");
        let (total, allowed, denied) = mw.audit_summary();
        assert_eq!(total, 2);
        assert_eq!(allowed, 1);
        assert_eq!(denied, 1);
    }

    #[test]
    fn test_helper_determinism() {
        assert_eq!(deterministic_hash("hello"), deterministic_hash("hello"));
        assert_eq!(
            compute_signature("POST", "/infer", 1000),
            compute_signature("POST", "/infer", 1000)
        );
        let t = generate_token_string("key1", 1000);
        assert_eq!(t, generate_token_string("key1", 1000));
        assert!(t.starts_with("tok_"));
    }
}
