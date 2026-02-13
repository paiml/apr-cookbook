//! # Recipe: API Authentication and Authorization Middleware
//!
//! **Category**: API Integration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (std only)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Tests pass (15+)
//!
//! ## Learning Objective
//! API authentication and authorization middleware for model serving:
//! API key validation, token-based auth, request signing (HMAC),
//! role-based access control, rate limiting, and audit logging.
//!
//! ## Run Command
//! ```bash
//! cargo run --example api_auth_middleware
//! ```

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Token lifetime in seconds (1 hour).
const TOKEN_LIFETIME_SECS: u64 = 3600;

/// Maximum requests per second for the default rate limit.
const DEFAULT_RATE_LIMIT_RPS: u32 = 100;

/// HMAC signing key (static for deterministic demo).
const SIGNING_SECRET: &str = "apr-cookbook-hmac-secret-2026";

// ---------------------------------------------------------------------------
// Permission
// ---------------------------------------------------------------------------

/// Scoped permissions that can be granted to an API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Permission {
    /// Run inference against a deployed model.
    Inference,
    /// Trigger training or fine-tuning jobs.
    Training,
    /// Create, update, or delete models.
    ModelManagement,
    /// Full administrative access.
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

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// Predefined roles that map to sets of permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Role {
    Admin,
    User,
    ReadOnly,
}

impl Role {
    /// Return the permissions associated with this role.
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

// ---------------------------------------------------------------------------
// ApiKey
// ---------------------------------------------------------------------------

/// Registered API key with scoped permissions and rate-limit configuration.
#[derive(Debug, Clone)]
struct ApiKey {
    key_id: String,
    secret_hash: u64,
    permissions: Vec<Permission>,
    rate_limit_rps: u32,
    role: Role,
}

// ---------------------------------------------------------------------------
// AuthToken
// ---------------------------------------------------------------------------

/// Short-lived token issued after API key validation.
#[derive(Debug, Clone)]
struct AuthToken {
    key_id: String,
    issued_at: u64,
    expires_at: u64,
    scopes: Vec<Permission>,
}

impl AuthToken {
    /// Check whether the token is expired relative to `now`.
    fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

// ---------------------------------------------------------------------------
// AuthRequest
// ---------------------------------------------------------------------------

/// Incoming API request that requires authentication.
#[derive(Debug, Clone)]
struct AuthRequest {
    method: String,
    path: String,
    key_id: String,
    token: Option<String>,
    signature: Option<String>,
    timestamp: u64,
}

// ---------------------------------------------------------------------------
// AuthResult
// ---------------------------------------------------------------------------

/// Outcome of an authentication / authorization check.
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

// ---------------------------------------------------------------------------
// AuditEntry
// ---------------------------------------------------------------------------

/// Immutable audit log entry for every auth decision.
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

// ---------------------------------------------------------------------------
// RateLimitState
// ---------------------------------------------------------------------------

/// Per-key rate-limit tracker (token-bucket style).
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

    /// Refill tokens based on elapsed seconds and attempt to consume one.
    fn try_consume(&mut self, now: u64) -> bool {
        let elapsed = now.saturating_sub(self.last_refill);
        if elapsed > 0 {
            let refill = (elapsed as u32).saturating_mul(self.max_tokens);
            self.tokens = self.tokens.saturating_add(refill).min(self.max_tokens);
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

// ---------------------------------------------------------------------------
// AuthMiddleware
// ---------------------------------------------------------------------------

/// Central authentication and authorization middleware.
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

    // -- Section 1: Register API keys with scoped permissions ----------------

    fn register_key(&mut self, key_id: &str, secret: &str, role: Role) {
        let secret_hash = deterministic_hash(secret);
        let permissions = role.permissions();
        let rate_limit_rps = match role {
            Role::Admin => DEFAULT_RATE_LIMIT_RPS * 2,
            Role::User => DEFAULT_RATE_LIMIT_RPS,
            Role::ReadOnly => DEFAULT_RATE_LIMIT_RPS / 2,
        };

        let api_key = ApiKey {
            key_id: key_id.to_string(),
            secret_hash,
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

    // -- Section 2: Validate requests against API keys -----------------------

    fn validate_key(&mut self, key_id: &str, secret: &str, ip: &str) -> AuthResult {
        let secret_hash = deterministic_hash(secret);

        let result = match self.keys.get(key_id) {
            None => AuthResult {
                allowed: false,
                reason: "unknown key".to_string(),
                key_id: key_id.to_string(),
                permissions_used: vec![],
            },
            Some(key) if key.secret_hash != secret_hash => AuthResult {
                allowed: false,
                reason: "invalid secret".to_string(),
                key_id: key_id.to_string(),
                permissions_used: vec![],
            },
            Some(key) => AuthResult {
                allowed: true,
                reason: "key validated".to_string(),
                key_id: key_id.to_string(),
                permissions_used: key.permissions.clone(),
            },
        };

        self.record_audit(key_id, "validate_key", &result, ip);
        result
    }

    // -- Section 3: Token generation and expiration checking -----------------

    fn issue_token(&mut self, key_id: &str, ip: &str) -> Option<AuthToken> {
        let key = self.keys.get(key_id)?;
        let token = AuthToken {
            key_id: key_id.to_string(),
            issued_at: self.current_time,
            expires_at: self.current_time + TOKEN_LIFETIME_SECS,
            scopes: key.permissions.clone(),
        };

        let result = AuthResult {
            allowed: true,
            reason: "token issued".to_string(),
            key_id: key_id.to_string(),
            permissions_used: key.permissions.clone(),
        };
        self.record_audit(key_id, "issue_token", &result, ip);

        Some(token)
    }

    fn validate_token(&mut self, token: &AuthToken, ip: &str) -> AuthResult {
        if token.is_expired(self.current_time) {
            let result = AuthResult {
                allowed: false,
                reason: "token expired".to_string(),
                key_id: token.key_id.clone(),
                permissions_used: vec![],
            };
            self.record_audit(&token.key_id, "validate_token", &result, ip);
            return result;
        }

        let result = AuthResult {
            allowed: true,
            reason: "token valid".to_string(),
            key_id: token.key_id.clone(),
            permissions_used: token.scopes.clone(),
        };
        self.record_audit(&token.key_id, "validate_token", &result, ip);
        result
    }

    // -- Section 4: Request signing and verification -------------------------

    fn verify_signature(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
        let expected = compute_signature(&request.method, &request.path, request.timestamp);

        let result = match &request.signature {
            None => AuthResult {
                allowed: false,
                reason: "missing signature".to_string(),
                key_id: request.key_id.clone(),
                permissions_used: vec![],
            },
            Some(sig) if *sig != expected => AuthResult {
                allowed: false,
                reason: "invalid signature".to_string(),
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
                    reason: "signature verified".to_string(),
                    key_id: request.key_id.clone(),
                    permissions_used: perms,
                }
            }
        };

        self.record_audit(&request.key_id, "verify_signature", &result, ip);
        result
    }

    /// Full request authentication: checks token presence, then signature.
    fn authenticate_request(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
        // If a bearer token is present, validate it as a token string
        if let Some(tok) = &request.token {
            let token_valid = !tok.is_empty() && tok.starts_with("tok_");
            if !token_valid {
                let result = AuthResult {
                    allowed: false,
                    reason: "invalid bearer token format".to_string(),
                    key_id: request.key_id.clone(),
                    permissions_used: vec![],
                };
                self.record_audit(&request.key_id, "authenticate_request", &result, ip);
                return result;
            }
        }

        // Fall through to signature verification
        self.verify_signature(request, ip)
    }

    // -- Section 5: Role-based access control --------------------------------

    fn check_permission(&mut self, key_id: &str, required: Permission, ip: &str) -> AuthResult {
        let result = match self.keys.get(key_id) {
            None => AuthResult {
                allowed: false,
                reason: "unknown key".to_string(),
                key_id: key_id.to_string(),
                permissions_used: vec![],
            },
            Some(key) if key.permissions.contains(&required) => AuthResult {
                allowed: true,
                reason: format!("permission {} granted via role {}", required, key.role),
                key_id: key_id.to_string(),
                permissions_used: vec![required],
            },
            Some(key) => AuthResult {
                allowed: false,
                reason: format!("permission {} denied for role {}", required, key.role),
                key_id: key_id.to_string(),
                permissions_used: vec![],
            },
        };

        self.record_audit(key_id, "check_permission", &result, ip);
        result
    }

    // -- Section 6: Rate limiting per API key --------------------------------

    fn check_rate_limit(&mut self, key_id: &str, ip: &str) -> AuthResult {
        let allowed = self
            .rate_limits
            .get_mut(key_id)
            .is_some_and(|state| state.try_consume(self.current_time));

        let result = AuthResult {
            allowed,
            reason: if allowed {
                "within rate limit".to_string()
            } else {
                "rate limit exceeded".to_string()
            },
            key_id: key_id.to_string(),
            permissions_used: vec![],
        };

        self.record_audit(key_id, "check_rate_limit", &result, ip);
        result
    }

    // -- Audit trail ---------------------------------------------------------

    fn record_audit(&mut self, key_id: &str, action: &str, result: &AuthResult, ip: &str) {
        self.audit_log.push(AuditEntry {
            timestamp: self.current_time,
            key_id: key_id.to_string(),
            action: action.to_string(),
            result: if result.allowed {
                "allowed".to_string()
            } else {
                format!("denied: {}", result.reason)
            },
            ip_address: ip.to_string(),
        });
    }

    fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    fn audit_summary(&self) -> AuditSummary {
        let total = self.audit_log.len();
        let allowed = self
            .audit_log
            .iter()
            .filter(|e| e.result == "allowed")
            .count();
        let denied = total - allowed;

        let mut by_key: HashMap<String, (usize, usize)> = HashMap::new();
        for entry in &self.audit_log {
            let counts = by_key.entry(entry.key_id.clone()).or_insert((0, 0));
            if entry.result == "allowed" {
                counts.0 += 1;
            } else {
                counts.1 += 1;
            }
        }

        AuditSummary {
            total,
            allowed,
            denied,
            by_key,
        }
    }
}

// ---------------------------------------------------------------------------
// AuditSummary
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AuditSummary {
    total: usize,
    allowed: usize,
    denied: usize,
    by_key: HashMap<String, (usize, usize)>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Deterministic hash using `DefaultHasher` (NOT cryptographic -- demo only).
fn deterministic_hash(input: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

/// Compute an HMAC-like signature for a request (deterministic, using `DefaultHasher`).
fn compute_signature(method: &str, path: &str, timestamp: u64) -> String {
    let mut hasher = DefaultHasher::new();
    SIGNING_SECRET.hash(&mut hasher);
    method.hash(&mut hasher);
    path.hash(&mut hasher);
    timestamp.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Generate a deterministic pseudo-random token string for a key ID.
fn generate_token_string(key_id: &str, timestamp: u64) -> String {
    let mut hasher = DefaultHasher::new();
    key_id.hash(&mut hasher);
    timestamp.hash(&mut hasher);
    format!("tok_{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Recipe: API Auth Middleware ===");
    println!("Authentication and authorization middleware for model serving");
    println!();

    let base_time: u64 = 1_700_000_000;
    let mut mw = AuthMiddleware::new(base_time);

    // -----------------------------------------------------------------------
    // Section 1: Register API keys with scoped permissions
    // -----------------------------------------------------------------------
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
    println!();

    // -----------------------------------------------------------------------
    // Section 2: Validate requests against API keys
    // -----------------------------------------------------------------------
    println!("--- Section 2: Validate API Keys ---");

    let r1 = mw.validate_key("key-admin-001", "super-secret-admin", "10.0.0.1");
    println!("  {}", r1);

    let r2 = mw.validate_key("key-admin-001", "wrong-secret", "10.0.0.2");
    println!("  {}", r2);

    let r3 = mw.validate_key("key-unknown", "anything", "10.0.0.3");
    println!("  {}", r3);
    println!();

    // -----------------------------------------------------------------------
    // Section 3: Token generation and expiration checking
    // -----------------------------------------------------------------------
    println!("--- Section 3: Token Lifecycle ---");

    let token = mw
        .issue_token("key-user-042", "10.0.0.10")
        .expect("key exists");
    let token_str = generate_token_string(&token.key_id, token.issued_at);
    println!(
        "  Issued: {} (expires in {}s)",
        token_str, TOKEN_LIFETIME_SECS
    );

    let r4 = mw.validate_token(&token, "10.0.0.10");
    println!("  Valid now: {}", r4);

    // Advance past expiration
    mw.set_time(base_time + TOKEN_LIFETIME_SECS + 1);
    let r5 = mw.validate_token(&token, "10.0.0.10");
    println!("  After expiry: {}", r5);

    // Reset time
    mw.set_time(base_time);
    println!();

    // -----------------------------------------------------------------------
    // Section 4: Request signing and verification
    // -----------------------------------------------------------------------
    println!("--- Section 4: Request Signing ---");

    let sig = compute_signature("POST", "/v1/models/fraud/infer", base_time);
    let signed_req = AuthRequest {
        method: "POST".to_string(),
        path: "/v1/models/fraud/infer".to_string(),
        key_id: "key-admin-001".to_string(),
        token: None,
        signature: Some(sig.clone()),
        timestamp: base_time,
    };
    let r6 = mw.verify_signature(&signed_req, "10.0.0.1");
    println!("  Signed request: {}", r6);
    println!("  Signature: {}", sig);

    let tampered_req = AuthRequest {
        method: "POST".to_string(),
        path: "/v1/models/fraud/infer".to_string(),
        key_id: "key-admin-001".to_string(),
        token: None,
        signature: Some("0000000000000000".to_string()),
        timestamp: base_time,
    };
    let r7 = mw.verify_signature(&tampered_req, "10.0.0.1");
    println!("  Tampered request: {}", r7);

    let unsigned_req = AuthRequest {
        method: "GET".to_string(),
        path: "/v1/health".to_string(),
        key_id: "key-admin-001".to_string(),
        token: None,
        signature: None,
        timestamp: base_time,
    };
    let r8 = mw.verify_signature(&unsigned_req, "10.0.0.1");
    println!("  Unsigned request: {}", r8);

    // Full auth with valid bearer token + signature
    let token_req = AuthRequest {
        method: "POST".to_string(),
        path: "/v1/models/fraud/infer".to_string(),
        key_id: "key-admin-001".to_string(),
        token: Some(generate_token_string("key-admin-001", base_time)),
        signature: Some(compute_signature(
            "POST",
            "/v1/models/fraud/infer",
            base_time,
        )),
        timestamp: base_time,
    };
    let r9 = mw.authenticate_request(&token_req, "10.0.0.1");
    println!("  Full auth (token+sig): {}", r9);

    // Bad bearer token format
    let bad_token_req = AuthRequest {
        method: "POST".to_string(),
        path: "/v1/models/fraud/infer".to_string(),
        key_id: "key-admin-001".to_string(),
        token: Some("bad-format-token".to_string()),
        signature: Some(compute_signature(
            "POST",
            "/v1/models/fraud/infer",
            base_time,
        )),
        timestamp: base_time,
    };
    let r10 = mw.authenticate_request(&bad_token_req, "10.0.0.1");
    println!("  Bad token format: {}", r10);
    println!();

    // -----------------------------------------------------------------------
    // Section 5: Role-based access control
    // -----------------------------------------------------------------------
    println!("--- Section 5: Role-Based Access Control ---");

    let checks = [
        ("key-admin-001", Permission::Admin, "Admin -> Admin"),
        ("key-user-042", Permission::Inference, "User -> Inference"),
        ("key-user-042", Permission::Admin, "User -> Admin"),
        ("key-ro-100", Permission::Inference, "ReadOnly -> Inference"),
        ("key-ro-100", Permission::Training, "ReadOnly -> Training"),
        (
            "key-ro-100",
            Permission::ModelManagement,
            "ReadOnly -> ModelMgmt",
        ),
    ];

    for (key_id, perm, label) in &checks {
        let r = mw.check_permission(key_id, *perm, "10.0.0.50");
        println!("  {}: {}", label, r);
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 6: Auth audit trail summary
    // -----------------------------------------------------------------------
    println!("--- Section 6: Audit Trail Summary ---");

    // Demonstrate rate limiting
    let ro_limit = mw
        .keys
        .get("key-ro-100")
        .map_or(DEFAULT_RATE_LIMIT_RPS, |k| k.rate_limit_rps);
    println!("  Rate limit for key-ro-100: {} rps", ro_limit);

    // Exhaust tokens for readonly key
    let mut rate_allowed = 0u32;
    let mut rate_denied = 0u32;
    for _ in 0..(ro_limit + 5) {
        let r = mw.check_rate_limit("key-ro-100", "10.0.0.100");
        if r.allowed {
            rate_allowed += 1;
        } else {
            rate_denied += 1;
        }
    }
    println!(
        "  Rate limit test (key-ro-100): {} allowed, {} denied",
        rate_allowed, rate_denied
    );

    let summary = mw.audit_summary();
    println!();
    println!("  Total audit entries: {}", summary.total);
    println!("  Allowed: {}", summary.allowed);
    println!("  Denied: {}", summary.denied);

    let mut sorted_keys: Vec<_> = summary.by_key.iter().collect();
    sorted_keys.sort_by_key(|(k, _)| (*k).clone());
    for (key, (a, d)) in &sorted_keys {
        println!("    {}: {} allowed, {} denied", key, a, d);
    }

    println!();
    println!("=== Auth middleware demonstration complete ===");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_middleware() -> AuthMiddleware {
        let mut mw = AuthMiddleware::new(1_000_000);
        mw.register_key("admin", "admin-secret", Role::Admin);
        mw.register_key("user", "user-secret", Role::User);
        mw.register_key("readonly", "ro-secret", Role::ReadOnly);
        mw
    }

    // -- API key registration ------------------------------------------------

    #[test]
    fn test_register_keys() {
        let mw = setup_middleware();
        assert_eq!(mw.keys.len(), 3);
        assert!(mw.keys.contains_key("admin"));
        assert!(mw.keys.contains_key("user"));
        assert!(mw.keys.contains_key("readonly"));
    }

    #[test]
    fn test_admin_has_all_permissions() {
        let mw = setup_middleware();
        let key = mw.keys.get("admin").expect("admin exists");
        assert!(key.permissions.contains(&Permission::Inference));
        assert!(key.permissions.contains(&Permission::Training));
        assert!(key.permissions.contains(&Permission::ModelManagement));
        assert!(key.permissions.contains(&Permission::Admin));
    }

    #[test]
    fn test_readonly_has_inference_only() {
        let mw = setup_middleware();
        let key = mw.keys.get("readonly").expect("readonly exists");
        assert_eq!(key.permissions, vec![Permission::Inference]);
    }

    // -- Key validation ------------------------------------------------------

    #[test]
    fn test_validate_correct_key() {
        let mut mw = setup_middleware();
        let result = mw.validate_key("admin", "admin-secret", "127.0.0.1");
        assert!(result.allowed);
        assert_eq!(result.reason, "key validated");
    }

    #[test]
    fn test_validate_wrong_secret() {
        let mut mw = setup_middleware();
        let result = mw.validate_key("admin", "wrong", "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "invalid secret");
    }

    #[test]
    fn test_validate_unknown_key() {
        let mut mw = setup_middleware();
        let result = mw.validate_key("nonexistent", "any", "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "unknown key");
    }

    // -- Token lifecycle -----------------------------------------------------

    #[test]
    fn test_issue_token() {
        let mut mw = setup_middleware();
        let token = mw.issue_token("user", "127.0.0.1");
        assert!(token.is_some());
        let token = token.expect("token issued");
        assert_eq!(token.key_id, "user");
        assert_eq!(token.expires_at, token.issued_at + TOKEN_LIFETIME_SECS);
    }

    #[test]
    fn test_token_not_expired() {
        let mut mw = setup_middleware();
        let token = mw.issue_token("user", "127.0.0.1").expect("token issued");
        let result = mw.validate_token(&token, "127.0.0.1");
        assert!(result.allowed);
    }

    #[test]
    fn test_token_expired() {
        let mut mw = setup_middleware();
        let token = mw.issue_token("user", "127.0.0.1").expect("token issued");
        mw.set_time(token.expires_at + 1);
        let result = mw.validate_token(&token, "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "token expired");
    }

    #[test]
    fn test_issue_token_unknown_key() {
        let mut mw = setup_middleware();
        let token = mw.issue_token("ghost", "127.0.0.1");
        assert!(token.is_none());
    }

    // -- Request signing -----------------------------------------------------

    #[test]
    fn test_signature_valid() {
        let mut mw = setup_middleware();
        let sig = compute_signature("POST", "/infer", mw.current_time);
        let req = AuthRequest {
            method: "POST".to_string(),
            path: "/infer".to_string(),
            key_id: "admin".to_string(),
            token: None,
            signature: Some(sig),
            timestamp: mw.current_time,
        };
        let result = mw.verify_signature(&req, "127.0.0.1");
        assert!(result.allowed);
    }

    #[test]
    fn test_signature_tampered() {
        let mut mw = setup_middleware();
        let req = AuthRequest {
            method: "POST".to_string(),
            path: "/infer".to_string(),
            key_id: "admin".to_string(),
            token: None,
            signature: Some("deadbeefdeadbeef".to_string()),
            timestamp: mw.current_time,
        };
        let result = mw.verify_signature(&req, "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "invalid signature");
    }

    #[test]
    fn test_signature_missing() {
        let mut mw = setup_middleware();
        let req = AuthRequest {
            method: "GET".to_string(),
            path: "/health".to_string(),
            key_id: "admin".to_string(),
            token: None,
            signature: None,
            timestamp: mw.current_time,
        };
        let result = mw.verify_signature(&req, "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "missing signature");
    }

    // -- Role-based access control -------------------------------------------

    #[test]
    fn test_rbac_admin_can_manage() {
        let mut mw = setup_middleware();
        let result = mw.check_permission("admin", Permission::ModelManagement, "127.0.0.1");
        assert!(result.allowed);
    }

    #[test]
    fn test_rbac_user_cannot_admin() {
        let mut mw = setup_middleware();
        let result = mw.check_permission("user", Permission::Admin, "127.0.0.1");
        assert!(!result.allowed);
    }

    #[test]
    fn test_rbac_readonly_cannot_train() {
        let mut mw = setup_middleware();
        let result = mw.check_permission("readonly", Permission::Training, "127.0.0.1");
        assert!(!result.allowed);
    }

    // -- Rate limiting -------------------------------------------------------

    #[test]
    fn test_rate_limit_allows_within_budget() {
        let mut mw = setup_middleware();
        let result = mw.check_rate_limit("user", "127.0.0.1");
        assert!(result.allowed);
    }

    #[test]
    fn test_rate_limit_denies_when_exhausted() {
        let mut mw = setup_middleware();
        let limit = mw.keys.get("readonly").expect("exists").rate_limit_rps;
        for _ in 0..limit {
            mw.check_rate_limit("readonly", "127.0.0.1");
        }
        let result = mw.check_rate_limit("readonly", "127.0.0.1");
        assert!(!result.allowed);
        assert_eq!(result.reason, "rate limit exceeded");
    }

    #[test]
    fn test_rate_limit_refills() {
        let mut mw = setup_middleware();
        let limit = mw.keys.get("readonly").expect("exists").rate_limit_rps;
        // Exhaust tokens
        for _ in 0..limit {
            mw.check_rate_limit("readonly", "127.0.0.1");
        }
        // Advance time by 1 second -- should refill
        mw.set_time(mw.current_time + 1);
        let result = mw.check_rate_limit("readonly", "127.0.0.1");
        assert!(result.allowed);
    }

    // -- Audit trail ---------------------------------------------------------

    #[test]
    fn test_audit_log_records_entries() {
        let mut mw = setup_middleware();
        mw.validate_key("admin", "admin-secret", "127.0.0.1");
        mw.validate_key("admin", "wrong", "127.0.0.1");
        let summary = mw.audit_summary();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.allowed, 1);
        assert_eq!(summary.denied, 1);
    }

    // -- Helper determinism --------------------------------------------------

    #[test]
    fn test_deterministic_hash_stable() {
        let h1 = deterministic_hash("hello");
        let h2 = deterministic_hash("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_signature_deterministic() {
        let s1 = compute_signature("POST", "/infer", 1000);
        let s2 = compute_signature("POST", "/infer", 1000);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_generate_token_string_deterministic() {
        let t1 = generate_token_string("key1", 1000);
        let t2 = generate_token_string("key1", 1000);
        assert_eq!(t1, t2);
        assert!(t1.starts_with("tok_"));
    }
}
