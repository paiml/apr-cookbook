//! # Contracts-Macros Recipe TTL Check
//!
//! Verify recipes have not exceeded their declared TTL since
//! creation. Returns sorted expired IDs and the count remaining
//! within TTL.
//!
//! Demonstrates the **CMM.176** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TLS certificate validity periods; DNS TTL semantics
//!  RFC 1035 §3.2.1.
//!
//! Run with: cargo run --example contracts_macros_recipe_ttl_check
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TtlVerdict {
    Ok {
        expired_ids: Vec<String>,
        valid_count: u32,
    },
    InvalidConfig,
}

/// Items: (id, age_days, ttl_days).
pub fn check(items: &[(&str, u32, u32)]) -> TtlVerdict {
    if items.is_empty() {
        return TtlVerdict::InvalidConfig;
    }
    for (_, _, ttl) in items {
        if *ttl == 0 {
            return TtlVerdict::InvalidConfig;
        }
    }
    let mut expired: Vec<String> = items
        .iter()
        .filter(|(_, age, ttl)| *age > *ttl)
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    expired.sort();
    let valid = (items.len() as u32) - (expired.len() as u32);
    TtlVerdict::Ok {
        expired_ids: expired,
        valid_count: valid,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_ttl_check")?;

    let items = [("r1", 30, 90), ("r2", 100, 90)];
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
    fn within_ttl_valid() {
        let v = check(&[("r", 30, 90)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert!(expired_ids.is_empty());
        }
    }

    #[test]
    fn over_ttl_expired() {
        let v = check(&[("r", 100, 90)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert_eq!(expired_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn at_ttl_valid() {
        let v = check(&[("r", 90, 90)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert!(expired_ids.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), TtlVerdict::InvalidConfig);
    }

    #[test]
    fn zero_ttl_rejected() {
        assert_eq!(check(&[("r", 5, 0)]), TtlVerdict::InvalidConfig);
    }

    #[test]
    fn valid_count_correct() {
        let v = check(&[("a", 30, 90), ("b", 200, 90), ("c", 60, 90)]);
        if let TtlVerdict::Ok { valid_count, .. } = v {
            assert_eq!(valid_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("r", 30, 90)]);
        let r2 = check(&[("r", 30, 90)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn expired_sorted() {
        let v = check(&[("zeta", 200, 90), ("alpha", 200, 90)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert_eq!(expired_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, u32, u32)> = (0..30).map(|_| ("r", 200, 90)).collect();
        let v = check(&items);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert_eq!(expired_ids.len(), 30);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", 200, 90)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert_eq!(expired_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn varied_ttls_per_item() {
        let v = check(&[("a", 50, 100), ("b", 50, 30)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert_eq!(expired_ids, vec!["b".to_string()]);
        }
    }

    #[test]
    fn no_expired_returns_empty() {
        let v = check(&[("a", 5, 30), ("b", 10, 30)]);
        if let TtlVerdict::Ok { expired_ids, .. } = v {
            assert!(expired_ids.is_empty());
        }
    }
}
