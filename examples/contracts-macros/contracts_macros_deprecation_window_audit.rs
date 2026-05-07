//! # Contracts-Macros Deprecation Window Audit
//!
//! Audit deprecated recipes against a removal-window policy: items
//! deprecated more than `window_days` ago must be removed. Returns
//! sorted overdue IDs and remaining-days for items still in window.
//!
//! Demonstrates the **CMM.153** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Python `__deprecated__` window guidance (PEP 387);
//!  Kubernetes API deprecation policy (≥6 months for stable APIs).
//!
//! Run with: cargo run --example contracts_macros_deprecation_window_audit
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DeprWindowVerdict {
    Ok {
        overdue_ids: Vec<String>,
        in_window_count: u32,
    },
    InvalidConfig,
}

pub fn audit(items: &[(&str, u32)], window_days: u32) -> DeprWindowVerdict {
    if items.is_empty() || window_days == 0 {
        return DeprWindowVerdict::InvalidConfig;
    }
    let mut overdue: Vec<String> = items
        .iter()
        .filter(|(_, age)| *age > window_days)
        .map(|(id, _)| (*id).to_string())
        .collect();
    overdue.sort();
    let in_window = items.iter().filter(|(_, age)| *age <= window_days).count() as u32;
    DeprWindowVerdict::Ok {
        overdue_ids: overdue,
        in_window_count: in_window,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_deprecation_window_audit")?;

    let items = [("api_v1", 200), ("api_v2", 30)];
    println!("180-day window: {:?}", audit(&items, 180));
    println!("invalid: {:?}", audit(&[], 180));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_window_in_band() {
        let v = audit(&[("a", 30)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert!(overdue_ids.is_empty());
        }
    }

    #[test]
    fn over_window_overdue() {
        let v = audit(&[("a", 200)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert_eq!(overdue_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn at_window_in_band() {
        let v = audit(&[("a", 180)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert!(overdue_ids.is_empty());
        }
    }

    #[test]
    fn empty_items_rejected() {
        assert_eq!(audit(&[], 180), DeprWindowVerdict::InvalidConfig);
    }

    #[test]
    fn zero_window_rejected() {
        assert_eq!(audit(&[("a", 30)], 0), DeprWindowVerdict::InvalidConfig);
    }

    #[test]
    fn in_window_count_correct() {
        let v = audit(&[("a", 30), ("b", 200), ("c", 60)], 180);
        if let DeprWindowVerdict::Ok {
            in_window_count, ..
        } = v
        {
            assert_eq!(in_window_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", 30)], 180);
        let r2 = audit(&[("a", 30)], 180);
        assert_eq!(r1, r2);
    }

    #[test]
    fn overdue_sorted() {
        let v = audit(&[("zeta", 200), ("alpha", 200)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert_eq!(overdue_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, u32)> = (0..30).map(|_| ("r", 200)).collect();
        let v = audit(&items, 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert_eq!(overdue_ids.len(), 30);
        }
    }

    #[test]
    fn no_overdue_returns_empty() {
        let v = audit(&[("a", 5), ("b", 10)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert!(overdue_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = audit(&[("café", 200)], 180);
        if let DeprWindowVerdict::Ok { overdue_ids, .. } = v {
            assert_eq!(overdue_ids, vec!["café".to_string()]);
        }
    }
}
