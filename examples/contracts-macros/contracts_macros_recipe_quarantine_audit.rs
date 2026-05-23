//! # Contracts-Macros Recipe Quarantine Audit
//!
//! Audit recipes flagged as quarantined (test-flaky, deprecated, or
//! known-broken). Returns sorted-by-severity quarantined IDs and a
//! count of recipes safe to release.
//!
//! Demonstrates the **CMM.150** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pytest `xfail`/`skip`; rustc `#[ignore]`; bazel
//!  `--test_filter=-quarantine` patterns.
//!
//! Run with: cargo run --example contracts_macros_recipe_quarantine_audit
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuarantineVerdict {
    Ok {
        quarantined_sorted: Vec<String>,
        release_safe_count: u32,
    },
    InvalidConfig,
}

/// `recipes` items: (id, severity 1..=10, is_quarantined)
pub fn audit(recipes: &[(&str, u8, bool)]) -> QuarantineVerdict {
    if recipes.is_empty() {
        return QuarantineVerdict::InvalidConfig;
    }
    for (_, sev, _) in recipes {
        if !(1..=10).contains(sev) {
            return QuarantineVerdict::InvalidConfig;
        }
    }
    let mut q: Vec<(&&str, u8)> = recipes
        .iter()
        .filter(|(_, _, qq)| *qq)
        .map(|(id, sev, _)| (id, *sev))
        .collect();
    // Sort: severity desc, id asc.
    q.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let quarantined_sorted: Vec<String> = q.iter().map(|(id, _)| (**id).to_string()).collect();
    let safe = recipes.iter().filter(|(_, _, qq)| !qq).count() as u32;
    QuarantineVerdict::Ok {
        quarantined_sorted,
        release_safe_count: safe,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_quarantine_audit")?;

    let r = [("r1", 5, false), ("r2", 9, true), ("r3", 3, true)];
    println!("audit: {:?}", audit(&r));
    println!("invalid: {:?}", audit(&[]));
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
    fn no_quarantined_empty_result() {
        let v = audit(&[("a", 5, false)]);
        if let QuarantineVerdict::Ok {
            quarantined_sorted, ..
        } = v
        {
            assert!(quarantined_sorted.is_empty());
        }
    }

    #[test]
    fn quarantined_listed() {
        let v = audit(&[("a", 5, true)]);
        if let QuarantineVerdict::Ok {
            quarantined_sorted, ..
        } = v
        {
            assert_eq!(quarantined_sorted, vec!["a".to_string()]);
        }
    }

    #[test]
    fn release_safe_count_correct() {
        let v = audit(&[("a", 5, false), ("b", 5, true), ("c", 5, false)]);
        if let QuarantineVerdict::Ok {
            release_safe_count, ..
        } = v
        {
            assert_eq!(release_safe_count, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), QuarantineVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_severity_zero() {
        assert_eq!(audit(&[("a", 0, true)]), QuarantineVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_severity_over_ten() {
        assert_eq!(audit(&[("a", 11, true)]), QuarantineVerdict::InvalidConfig);
    }

    #[test]
    fn higher_severity_first() {
        let v = audit(&[("low", 2, true), ("high", 9, true)]);
        if let QuarantineVerdict::Ok {
            quarantined_sorted, ..
        } = v
        {
            assert_eq!(quarantined_sorted[0], "high");
            assert_eq!(quarantined_sorted[1], "low");
        }
    }

    #[test]
    fn equal_severity_alphabetical() {
        let v = audit(&[("zeta", 5, true), ("alpha", 5, true)]);
        if let QuarantineVerdict::Ok {
            quarantined_sorted, ..
        } = v
        {
            assert_eq!(
                quarantined_sorted,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", 5, true)]);
        let r2 = audit(&[("a", 5, true)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_recipes_handled() {
        let r: Vec<(&str, u8, bool)> = (0..30).map(|_| ("r", 5, false)).collect();
        let v = audit(&r);
        if let QuarantineVerdict::Ok {
            release_safe_count, ..
        } = v
        {
            assert_eq!(release_safe_count, 30);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = audit(&[("café", 5, true)]);
        if let QuarantineVerdict::Ok {
            quarantined_sorted, ..
        } = v
        {
            assert_eq!(quarantined_sorted, vec!["café".to_string()]);
        }
    }
}
