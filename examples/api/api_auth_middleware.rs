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

const TOKEN_LIFETIME_SECS: u64 = 3600;
const DEFAULT_RATE_LIMIT_RPS: u32 = 100;
const SIGNING_SECRET: &str = "apr-cookbook-hmac-secret-2026";

// ---- Domain Types -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Permission {
    Inference,
    Training,
    ModelManagement,
    Admin,
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference => write!(f, "Inference"),
            Self::Training => write!(f, "Training"),
            Self::ModelManagement => write!(f, "ModelManagement"),
            Self::Admin => write!(f, "Admin"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Role {
    Admin,
    User,
    ReadOnly,
}

impl Role {
    fn permissions(self) -> Vec<Permission> {
        match self {
            Self::Admin => vec![
                Permission::Inference,
                Permission::Training,
                Permission::ModelManagement,
                Permission::Admin,
            ],
            Self::User => vec![Permission::Inference, Permission::Training],
            Self::ReadOnly => vec![Permission::Inference],
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admin => write!(f, "Admin"),
            Self::User => write!(f, "User"),
            Self::ReadOnly => write!(f, "ReadOnly"),
        }
    }
}

#[derive(Debug, Clone)]
struct ApiKey {
    key_id: String,
    secret_hash: u64,
    permissions: Vec<Permission>,
    rate_limit_rps: u32,
    role: Role,
}

#[derive(Debug, Clone)]
struct AuthToken {
    key_id: String,
    #[allow(dead_code)]
    issued_at: u64,
    expires_at: u64,
    scopes: Vec<Permission>,
}

impl AuthToken {
    fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

#[derive(Debug, Clone)]
struct AuthRequest {
    method: String,
    path: String,
    key_id: String,
    token: Option<String>,
    signature: Option<String>,
    timestamp: u64,
}

#[derive(Debug, Clone)]
struct AuthResult {
    allowed: bool,
    reason: String,
    key_id: String,
    permissions_used: Vec<Permission>,
}

impl fmt::Display for AuthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.allowed { "ALLOWED" } else { "DENIED" };
        let perms: Vec<String> = self
            .permissions_used
            .iter()
            .map(ToString::to_string)
            .collect();
        write!(
            f,
            "[{}] key={} reason={} perms=[{}]",
            status,
            self.key_id,
            self.reason,
            perms.join(", ")
        )
    }
}

#[derive(Debug, Clone)]
struct AuditEntry {
    timestamp: u64,
    key_id: String,
    action: String,
    result: String,
    ip_address: String,
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "t={} key={} action={} result={} ip={}",
            self.timestamp, self.key_id, self.action, self.result, self.ip_address
        )
    }
}

// ---- Rate Limiting ----------------------------------------------------------

#[derive(Debug, Clone)]
struct RateLimitState {
    tokens: u32,
    max_tokens: u32,
    last_refill: u64,
}

impl RateLimitState {
    fn new(max_tokens: u32, now: u64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            last_refill: now,
        }
    }
    fn try_consume(&mut self, now: u64) -> bool {
        let elapsed = now.saturating_sub(self.last_refill);
        if elapsed > 0 {
            self.tokens = self
                .tokens
                .saturating_add((elapsed as u32).saturating_mul(self.max_tokens))
                .min(self.max_tokens);
            self.last_refill = now;
        }
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }
}

// ---- AuthMiddleware ---------------------------------------------------------

struct AuthMiddleware {
    keys: HashMap<String, ApiKey>,
    rate_limits: HashMap<String, RateLimitState>,
    audit_log: Vec<AuditEntry>,
    current_time: u64,
}

impl AuthMiddleware {
    fn new(start_time: u64) -> Self {
        Self {
            keys: HashMap::new(),
            rate_limits: HashMap::new(),
            audit_log: Vec::new(),
            current_time: start_time,
        }
    }

    fn register_key(&mut self, key_id: &str, secret: &str, role: Role) {
        let permissions = role.permissions();
        let rate_limit_rps = match role {
            Role::Admin => DEFAULT_RATE_LIMIT_RPS * 2,
            Role::User => DEFAULT_RATE_LIMIT_RPS,
            Role::ReadOnly => DEFAULT_RATE_LIMIT_RPS / 2,
        };
        let api_key = ApiKey {
            key_id: key_id.to_string(),
            secret_hash: deterministic_hash(secret),
            permissions,
            rate_limit_rps,
            role,
        };
        self.rate_limits.insert(
            key_id.to_string(),
            RateLimitState::new(rate_limit_rps, self.current_time),
        );
        self.keys.insert(key_id.to_string(), api_key);
    }

    fn validate_key(&mut self, key_id: &str, secret: &str, ip: &str) -> AuthResult {
        let hash = deterministic_hash(secret);
        let result = match self.keys.get(key_id) {
            None => AuthResult {
                allowed: false,
                reason: "unknown key".into(),
                key_id: key_id.into(),
                permissions_used: vec![],
            },
            Some(k) if k.secret_hash != hash => AuthResult {
                allowed: false,
                reason: "invalid secret".into(),
                key_id: key_id.into(),
                permissions_used: vec![],
            },
            Some(k) => AuthResult {
                allowed: true,
                reason: "key validated".into(),
                key_id: key_id.into(),
                permissions_used: k.permissions.clone(),
            },
        };
        self.record_audit(key_id, "validate_key", &result, ip);
        result
    }

    fn issue_token(&mut self, key_id: &str, ip: &str) -> Option<AuthToken> {
        let key = self.keys.get(key_id)?;
        let token = AuthToken {
            key_id: key_id.into(),
            issued_at: self.current_time,
            expires_at: self.current_time + TOKEN_LIFETIME_SECS,
            scopes: key.permissions.clone(),
        };
        let result = AuthResult {
            allowed: true,
            reason: "token issued".into(),
            key_id: key_id.into(),
            permissions_used: key.permissions.clone(),
        };
        self.record_audit(key_id, "issue_token", &result, ip);
        Some(token)
    }

    fn validate_token(&mut self, token: &AuthToken, ip: &str) -> AuthResult {
        let result = if token.is_expired(self.current_time) {
            AuthResult {
                allowed: false,
                reason: "token expired".into(),
                key_id: token.key_id.clone(),
                permissions_used: vec![],
            }
        } else {
            AuthResult {
                allowed: true,
                reason: "token valid".into(),
                key_id: token.key_id.clone(),
                permissions_used: token.scopes.clone(),
            }
        };
        self.record_audit(&token.key_id, "validate_token", &result, ip);
        result
    }

    fn verify_signature(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
        let expected = compute_signature(&request.method, &request.path, request.timestamp);
        let result = match &request.signature {
            None => AuthResult {
                allowed: false,
                reason: "missing signature".into(),
                key_id: request.key_id.clone(),
                permissions_used: vec![],
            },
            Some(sig) if *sig != expected => AuthResult {
                allowed: false,
                reason: "invalid signature".into(),
                key_id: request.key_id.clone(),
                permissions_used: vec![],
            },
            Some(_) => {
                let perms = self
                    .keys
                    .get(&request.key_id)
                    .map(|k| k.permissions.clone())
                    .unwrap_or_default();
                AuthResult {
                    allowed: true,
                    reason: "signature verified".into(),
                    key_id: request.key_id.clone(),
                    permissions_used: perms,
                }
            }
        };
        self.record_audit(&request.key_id, "verify_signature", &result, ip);
        result
    }

    fn authenticate_request(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
        if let Some(tok) = &request.token {
            if !tok.starts_with("tok_") || tok.is_empty() {
                let result = AuthResult {
                    allowed: false,
                    reason: "invalid bearer token format".into(),
                    key_id: request.key_id.clone(),
                    permissions_used: vec![],
                };
                self.record_audit(&request.key_id, "authenticate_request", &result, ip);
                return result;
            }
        }
        self.verify_signature(request, ip)
    }

    fn check_permission(&mut self, key_id: &str, required: Permission, ip: &str) -> AuthResult {
        let result = match self.keys.get(key_id) {
            None => AuthResult {
                allowed: false,
                reason: "unknown key".into(),
                key_id: key_id.into(),
                permissions_used: vec![],
            },
            Some(k) if k.permissions.contains(&required) => AuthResult {
                allowed: true,
                reason: format!("permission {} granted via role {}", required, k.role),
                key_id: key_id.into(),
                permissions_used: vec![required],
            },
            Some(k) => AuthResult {
                allowed: false,
                reason: format!("permission {} denied for role {}", required, k.role),
                key_id: key_id.into(),
                permissions_used: vec![],
            },
        };
        self.record_audit(key_id, "check_permission", &result, ip);
        result
    }

    fn check_rate_limit(&mut self, key_id: &str, ip: &str) -> AuthResult {
        let allowed = self
            .rate_limits
            .get_mut(key_id)
            .is_some_and(|s| s.try_consume(self.current_time));
        let result = AuthResult {
            allowed,
            reason: if allowed {
                "within rate limit".into()
            } else {
                "rate limit exceeded".into()
            },
            key_id: key_id.into(),
            permissions_used: vec![],
        };
        self.record_audit(key_id, "check_rate_limit", &result, ip);
        result
    }

    fn record_audit(&mut self, key_id: &str, action: &str, result: &AuthResult, ip: &str) {
        self.audit_log.push(AuditEntry {
            timestamp: self.current_time,
            key_id: key_id.into(),
            action: action.into(),
            result: if result.allowed {
                "allowed".into()
            } else {
                format!("denied: {}", result.reason)
            },
            ip_address: ip.into(),
        });
    }

    fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    fn audit_summary(&self) -> (usize, usize, usize) {
        let total = self.audit_log.len();
        let allowed = self
            .audit_log
            .iter()
            .filter(|e| e.result == "allowed")
            .count();
        (total, allowed, total - allowed)
    }
}

// ---- Helpers ----------------------------------------------------------------

fn deterministic_hash(input: &str) -> u64 {
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    h.finish()
}

fn compute_signature(method: &str, path: &str, timestamp: u64) -> String {
    let mut h = DefaultHasher::new();
    SIGNING_SECRET.hash(&mut h);
    method.hash(&mut h);
    path.hash(&mut h);
    timestamp.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn generate_token_string(key_id: &str, timestamp: u64) -> String {
    let mut h = DefaultHasher::new();
    key_id.hash(&mut h);
    timestamp.hash(&mut h);
    format!("tok_{:016x}", h.finish())
}

// ---- Main -------------------------------------------------------------------

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
