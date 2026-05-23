//! # Contracts-Macros Recipe Owner Email Valid
//!
//! Validate recipe owner email addresses against a basic structural
//! check (one '@', non-empty local + domain, dot in domain). Returns
//! sorted invalid IDs.
//!
//! Demonstrates the **CMM.191** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 5321 §4.1.2 mailbox syntax (simplified); HTML5
//!  `<input type="email">` lenient regex.
//!
//! Run with: cargo run --example contracts_macros_recipe_owner_email_valid
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EmailValidVerdict {
    Ok {
        invalid_ids: Vec<String>,
        valid_count: u32,
    },
    InvalidConfig,
}

pub fn check(items: &[(&str, &str)]) -> EmailValidVerdict {
    if items.is_empty() {
        return EmailValidVerdict::InvalidConfig;
    }
    let mut invalid: Vec<String> = items
        .iter()
        .filter(|(_, email)| !is_valid_email(email))
        .map(|(id, _)| (*id).to_string())
        .collect();
    invalid.sort();
    let valid_count = items.len() as u32 - invalid.len() as u32;
    EmailValidVerdict::Ok {
        invalid_ids: invalid,
        valid_count,
    }
}

fn is_valid_email(email: &str) -> bool {
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return false;
    }
    let (local, domain) = (parts[0], parts[1]);
    if local.is_empty() || domain.is_empty() {
        return false;
    }
    if !domain.contains('.') {
        return false;
    }
    // Domain TLD must be non-empty (no trailing dot).
    if domain.ends_with('.') {
        return false;
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_owner_email_valid")?;

    let items = [
        ("r1", "alice@example.com"),
        ("r2", "no-at-sign.com"),
        ("r3", "@nolocal.com"),
    ];
    println!("check: {:?}", check(&items));
    println!("invalid: {:?}", check(&[]));
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
    fn valid_email_no_offender() {
        let v = check(&[("r", "alice@example.com")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert!(invalid_ids.is_empty());
        }
    }

    #[test]
    fn missing_at_invalid() {
        let v = check(&[("r", "noatpart.com")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn missing_local_invalid() {
        let v = check(&[("r", "@example.com")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn missing_domain_invalid() {
        let v = check(&[("r", "alice@")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn missing_dot_in_domain_invalid() {
        let v = check(&[("r", "alice@localhost")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), EmailValidVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("r", "x@y.z")]);
        let r2 = check(&[("r", "x@y.z")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn double_at_invalid() {
        let v = check(&[("r", "a@b@c.com")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn invalid_sorted() {
        let v = check(&[("zeta", "bad"), ("alpha", "bad")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn valid_count_correct() {
        let v = check(&[("a", "x@y.z"), ("b", "bad"), ("c", "good@e.x")]);
        if let EmailValidVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 2);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("r", "bad")).collect();
        let v = check(&items);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids.len(), 30);
        }
    }

    #[test]
    fn trailing_dot_invalid() {
        let v = check(&[("r", "alice@example.")]);
        if let EmailValidVerdict::Ok { invalid_ids, .. } = v {
            assert_eq!(invalid_ids, vec!["r".to_string()]);
        }
    }
}
