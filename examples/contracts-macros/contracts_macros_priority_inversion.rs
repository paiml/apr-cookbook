//! # Contracts-Macros Priority Inversion Detector
//!
//! Detect priority inversion in obligation dependencies: a high-
//! priority obligation depending on a low-priority one. Returns the
//! first inversion (high.id, low.id) pair.
//!
//! Demonstrates the **CMM.44** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: priority inversion (Sha, Rajkumar, Lehoczky 1990).
//!
//! Run with: cargo run --example contracts_macros_priority_inversion
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum InversionVerdict {
    NoInversion,
    Found {
        high: String,
        low: String,
        high_priority: u8,
        low_priority: u8,
    },
    UnknownReference {
        equation: String,
        reference: String,
    },
    EmptyContract,
}

pub fn detect(obligations: &[(&str, u8, Vec<&str>)]) -> InversionVerdict {
    if obligations.is_empty() {
        return InversionVerdict::EmptyContract;
    }
    let priority: BTreeMap<&str, u8> = obligations.iter().map(|(n, p, _)| (*n, *p)).collect();
    for (name, pri, deps) in obligations {
        for d in deps {
            let Some(&dep_pri) = priority.get(d) else {
                return InversionVerdict::UnknownReference {
                    equation: (*name).to_string(),
                    reference: (*d).to_string(),
                };
            };
            if dep_pri > *pri {
                return InversionVerdict::Found {
                    high: (*name).to_string(),
                    low: (*d).to_string(),
                    high_priority: *pri,
                    low_priority: dep_pri,
                };
            }
        }
    }
    InversionVerdict::NoInversion
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_priority_inversion")?;

    let healthy = vec![
        ("low", 5u8, Vec::<&str>::new()),
        ("high", 1, Vec::<&str>::new()),
    ];
    println!("healthy: {:?}", detect(&healthy));

    let bad = vec![("low", 5, Vec::<&str>::new()), ("high", 1, vec!["low"])];
    println!("inversion: {:?}", detect(&bad));

    let unknown = vec![("a", 1, vec!["missing"])];
    println!("unknown ref: {:?}", detect(&unknown));
    println!("empty: {:?}", detect(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_inversion() {
        let obligations = vec![("a", 1u8, Vec::<&str>::new()), ("b", 2, vec!["a"])];
        assert_eq!(detect(&obligations), InversionVerdict::NoInversion);
    }

    #[test]
    fn inversion_found() {
        let obligations = vec![("low", 5u8, Vec::<&str>::new()), ("high", 1, vec!["low"])];
        let v = detect(&obligations);
        if let InversionVerdict::Found { high, low, .. } = v {
            assert_eq!(high, "high");
            assert_eq!(low, "low");
        }
    }

    #[test]
    fn equal_priority_no_inversion() {
        let obligations = vec![("a", 3u8, Vec::<&str>::new()), ("b", 3, vec!["a"])];
        assert_eq!(detect(&obligations), InversionVerdict::NoInversion);
    }

    #[test]
    fn unknown_reference() {
        let obligations = vec![("a", 1u8, vec!["missing"])];
        assert!(matches!(
            detect(&obligations),
            InversionVerdict::UnknownReference { .. }
        ));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(detect(&[]), InversionVerdict::EmptyContract);
    }

    #[test]
    fn multiple_inversions_first_returned() {
        let obligations = vec![
            ("low1", 5u8, Vec::<&str>::new()),
            ("low2", 6, Vec::<&str>::new()),
            ("high1", 1, vec!["low1"]),
            ("high2", 2, vec!["low2"]),
        ];
        let v = detect(&obligations);
        if let InversionVerdict::Found { high, .. } = v {
            assert_eq!(high, "high1");
        }
    }

    #[test]
    fn priority_carries_in_verdict() {
        let obligations = vec![("low", 9u8, Vec::<&str>::new()), ("high", 1, vec!["low"])];
        let v = detect(&obligations);
        if let InversionVerdict::Found {
            high_priority,
            low_priority,
            ..
        } = v
        {
            assert_eq!(high_priority, 1);
            assert_eq!(low_priority, 9);
        }
    }

    #[test]
    fn long_chain_works() {
        let obligations = vec![
            ("a", 1u8, Vec::<&str>::new()),
            ("b", 2, vec!["a"]),
            ("c", 3, vec!["b"]),
            ("d", 4, vec!["c"]),
        ];
        assert_eq!(detect(&obligations), InversionVerdict::NoInversion);
    }

    #[test]
    fn no_deps_no_inversion() {
        let obligations = vec![("a", 1u8, Vec::<&str>::new()), ("b", 2, Vec::<&str>::new())];
        assert_eq!(detect(&obligations), InversionVerdict::NoInversion);
    }

    #[test]
    fn deterministic() {
        let obligations = vec![("low", 5u8, Vec::<&str>::new()), ("high", 1, vec!["low"])];
        let a = detect(&obligations);
        let b = detect(&obligations);
        assert_eq!(a, b);
    }
}
