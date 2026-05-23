//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

pub const TOKEN_LIFETIME_SECS: u64 = 3600;
pub const DEFAULT_RATE_LIMIT_RPS: u32 = 100;
pub const SIGNING_SECRET: &str = "apr-cookbook-hmac-secret-2026";

// ---- Domain Types -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Permission {
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
pub enum Role {
    Admin,
    User,
    ReadOnly,
}

impl Role {
    pub fn permissions(self) -> Vec<Permission> {
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
pub struct ApiKey {
    pub key_id: String,
    pub secret_hash: u64,
    pub permissions: Vec<Permission>,
    pub rate_limit_rps: u32,
    pub role: Role,
}

#[derive(Debug, Clone)]
pub struct AuthToken {
    pub key_id: String,
    #[allow(dead_code)]
    pub issued_at: u64,
    pub expires_at: u64,
    pub scopes: Vec<Permission>,
}

impl AuthToken {
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

#[derive(Debug, Clone)]
pub struct AuthRequest {
    pub method: String,
    pub path: String,
    pub key_id: String,
    pub token: Option<String>,
    pub signature: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct AuthResult {
    pub allowed: bool,
    pub reason: String,
    pub key_id: String,
    pub permissions_used: Vec<Permission>,
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
pub struct AuditEntry {
    pub timestamp: u64,
    pub key_id: String,
    pub action: String,
    pub result: String,
    pub ip_address: String,
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
pub struct RateLimitState {
    pub tokens: u32,
    pub max_tokens: u32,
    pub last_refill: u64,
}

impl RateLimitState {
    pub fn new(max_tokens: u32, now: u64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            last_refill: now,
        }
    }
    pub fn try_consume(&mut self, now: u64) -> bool {
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

pub struct AuthMiddleware {
    pub keys: HashMap<String, ApiKey>,
    pub rate_limits: HashMap<String, RateLimitState>,
    pub audit_log: Vec<AuditEntry>,
    pub current_time: u64,
}

impl AuthMiddleware {
    pub fn new(start_time: u64) -> Self {
        Self {
            keys: HashMap::new(),
            rate_limits: HashMap::new(),
            audit_log: Vec::new(),
            current_time: start_time,
        }
    }

    pub fn register_key(&mut self, key_id: &str, secret: &str, role: Role) {
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

    pub fn validate_key(&mut self, key_id: &str, secret: &str, ip: &str) -> AuthResult {
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

    pub fn issue_token(&mut self, key_id: &str, ip: &str) -> Option<AuthToken> {
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

    pub fn validate_token(&mut self, token: &AuthToken, ip: &str) -> AuthResult {
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

    pub fn verify_signature(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
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

    pub fn authenticate_request(&mut self, request: &AuthRequest, ip: &str) -> AuthResult {
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

    pub fn check_permission(&mut self, key_id: &str, required: Permission, ip: &str) -> AuthResult {
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

    pub fn check_rate_limit(&mut self, key_id: &str, ip: &str) -> AuthResult {
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

    pub fn record_audit(&mut self, key_id: &str, action: &str, result: &AuthResult, ip: &str) {
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

    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    pub fn audit_summary(&self) -> (usize, usize, usize) {
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

pub fn deterministic_hash(input: &str) -> u64 {
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    h.finish()
}

pub fn compute_signature(method: &str, path: &str, timestamp: u64) -> String {
    let mut h = DefaultHasher::new();
    SIGNING_SECRET.hash(&mut h);
    method.hash(&mut h);
    path.hash(&mut h);
    timestamp.hash(&mut h);
    format!("{:016x}", h.finish())
}

pub fn generate_token_string(key_id: &str, timestamp: u64) -> String {
    let mut h = DefaultHasher::new();
    key_id.hash(&mut h);
    timestamp.hash(&mut h);
    format!("tok_{:016x}", h.finish())
}

// ---- Main -------------------------------------------------------------------
