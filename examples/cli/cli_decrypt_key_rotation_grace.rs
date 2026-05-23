//! # apr decrypt --key-id — Key Rotation Grace Window Validator
//!
//! Models encrypted with key K_n must remain decryptable for a grace
//! window after K_{n+1} ships. Default grace = 90 days; emergency
//! rotation (compromise) sets grace = 0. This recipe builds the
//! validator over (model_age_days, current_key_id, model_key_id,
//! grace_days) → Allowed/Expired/Compromised.
//!
//! Demonstrates the **DEC.6** recipe for PMAT-121 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEC-001 + NIST SP 800-57 (key management)
//!
//! Run with: cargo run --example cli_decrypt_key_rotation_grace
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_GRACE_DAYS: u32 = 90;

#[derive(Debug, PartialEq)]
pub enum KeyVerdict {
    Allowed,
    Expired { age_days: u32, grace_days: u32 },
    KeyCompromisedRevoke,
    UnknownKeyId,
}

#[derive(Debug, Clone, Copy)]
pub struct KeyMetadata {
    pub current_key_id: u32,
    pub revoked_keys: u32, // bitmap of revoked key IDs (low 32)
    pub grace_days: u32,
}

impl KeyMetadata {
    pub fn default_grace() -> Self {
        Self {
            current_key_id: 1,
            revoked_keys: 0,
            grace_days: DEFAULT_GRACE_DAYS,
        }
    }

    pub fn is_revoked(self, key_id: u32) -> bool {
        if key_id >= 32 {
            return false;
        }
        (self.revoked_keys >> key_id) & 1 == 1
    }
}

pub fn validate(model_key_id: u32, model_age_days: u32, meta: KeyMetadata) -> KeyVerdict {
    if model_key_id == 0 || model_key_id > meta.current_key_id {
        return KeyVerdict::UnknownKeyId;
    }
    if meta.is_revoked(model_key_id) {
        return KeyVerdict::KeyCompromisedRevoke;
    }
    if model_key_id == meta.current_key_id {
        return KeyVerdict::Allowed;
    }
    if meta.grace_days > 0 && model_age_days < meta.grace_days {
        KeyVerdict::Allowed
    } else {
        KeyVerdict::Expired {
            age_days: model_age_days,
            grace_days: meta.grace_days,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_key_rotation_grace")?;

    let meta = KeyMetadata {
        current_key_id: 3,
        revoked_keys: 1 << 1, // key id 1 revoked
        grace_days: 90,
    };
    let cases = [
        (3u32, 10u32), // current key, fresh
        (2, 30),       // previous, within grace
        (2, 200),      // previous, past grace
        (1, 5),        // revoked
        (99, 0),       // unknown
    ];
    for (kid, age) in cases {
        println!("key={kid} age={age}d  →  {:?}", validate(kid, age, meta));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(current: u32, grace: u32) -> KeyMetadata {
        KeyMetadata {
            current_key_id: current,
            revoked_keys: 0,
            grace_days: grace,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn current_key_always_allowed() {
        assert_eq!(validate(3, 0, meta(3, 90)), KeyVerdict::Allowed);
        assert_eq!(validate(3, 9999, meta(3, 90)), KeyVerdict::Allowed);
    }

    #[test]
    fn previous_key_within_grace_allowed() {
        assert_eq!(validate(2, 30, meta(3, 90)), KeyVerdict::Allowed);
    }

    #[test]
    fn previous_key_past_grace_expired() {
        let v = validate(2, 200, meta(3, 90));
        assert!(matches!(v, KeyVerdict::Expired { .. }));
    }

    #[test]
    fn at_grace_boundary_expired() {
        // age == grace_days → expired (exclusive boundary). The key
        // must be replaced strictly before the grace window closes.
        let v = validate(2, 90, meta(3, 90));
        assert!(matches!(v, KeyVerdict::Expired { .. }));
    }

    #[test]
    fn just_inside_grace_allowed() {
        // age == grace_days - 1 → still allowed.
        assert_eq!(validate(2, 89, meta(3, 90)), KeyVerdict::Allowed);
    }

    #[test]
    fn revoked_key_compromised() {
        let m = KeyMetadata {
            current_key_id: 3,
            revoked_keys: 1 << 1,
            grace_days: 90,
        };
        assert_eq!(validate(1, 5, m), KeyVerdict::KeyCompromisedRevoke);
    }

    #[test]
    fn revoked_takes_priority_over_grace() {
        // Even if within grace window, revoked key fails.
        let m = KeyMetadata {
            current_key_id: 3,
            revoked_keys: 1 << 2,
            grace_days: 90,
        };
        assert_eq!(validate(2, 1, m), KeyVerdict::KeyCompromisedRevoke);
    }

    #[test]
    fn unknown_key_id_rejected() {
        // Key ID > current is unknown.
        assert_eq!(validate(99, 0, meta(3, 90)), KeyVerdict::UnknownKeyId);
    }

    #[test]
    fn key_id_zero_rejected() {
        assert_eq!(validate(0, 0, meta(3, 90)), KeyVerdict::UnknownKeyId);
    }

    #[test]
    fn zero_grace_emergency_rotation() {
        // Emergency mode: grace=0 → only current key allowed.
        let v = validate(2, 0, meta(3, 0));
        assert!(matches!(v, KeyVerdict::Expired { grace_days: 0, .. }));
    }
}
