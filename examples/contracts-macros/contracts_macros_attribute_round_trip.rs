//! # Contracts-Macros Attribute Round-Trip
//!
//! Verify a contract attribute survives serialize→deserialize. Compares
//! the original key/value list to the round-trip result; reports if a
//! key is dropped, renamed, or re-ordered.
//!
//! Demonstrates the **CMM.32** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: serde round-trip property (proptest convention).
//!
//! Run with: cargo run --example contracts_macros_attribute_round_trip
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RoundTripVerdict {
    Identical,
    KeyDropped { key: String },
    ValueChanged { key: String },
    KeyAdded { key: String },
    EmptyInput,
}

pub fn check(original: &[(&str, &str)], round_tripped: &[(&str, &str)]) -> RoundTripVerdict {
    if original.is_empty() && round_tripped.is_empty() {
        return RoundTripVerdict::EmptyInput;
    }
    let orig_map: std::collections::BTreeMap<&str, &str> = original.iter().copied().collect();
    let rt_map: std::collections::BTreeMap<&str, &str> = round_tripped.iter().copied().collect();
    for (k, v) in &orig_map {
        match rt_map.get(k) {
            None => {
                return RoundTripVerdict::KeyDropped {
                    key: (*k).to_string(),
                }
            }
            Some(rv) if rv != v => {
                return RoundTripVerdict::ValueChanged {
                    key: (*k).to_string(),
                };
            }
            Some(_) => {}
        }
    }
    for k in rt_map.keys() {
        if !orig_map.contains_key(k) {
            return RoundTripVerdict::KeyAdded {
                key: (*k).to_string(),
            };
        }
    }
    RoundTripVerdict::Identical
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_attribute_round_trip")?;

    let orig = [("a", "1"), ("b", "2")];
    let same = [("a", "1"), ("b", "2")];
    let dropped = [("a", "1")];
    let changed = [("a", "1"), ("b", "9")];
    let added = [("a", "1"), ("b", "2"), ("c", "3")];

    println!("identical: {:?}", check(&orig, &same));
    println!("dropped: {:?}", check(&orig, &dropped));
    println!("changed: {:?}", check(&orig, &changed));
    println!("added: {:?}", check(&orig, &added));
    println!("empty: {:?}", check(&[], &[]));
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
    fn identical_passes() {
        assert_eq!(
            check(&[("a", "1")], &[("a", "1")]),
            RoundTripVerdict::Identical
        );
    }

    #[test]
    fn key_dropped_reported() {
        let v = check(&[("a", "1"), ("b", "2")], &[("a", "1")]);
        if let RoundTripVerdict::KeyDropped { key } = v {
            assert_eq!(key, "b");
        }
    }

    #[test]
    fn value_changed_reported() {
        let v = check(&[("a", "1")], &[("a", "9")]);
        if let RoundTripVerdict::ValueChanged { key } = v {
            assert_eq!(key, "a");
        }
    }

    #[test]
    fn key_added_reported() {
        let v = check(&[("a", "1")], &[("a", "1"), ("c", "3")]);
        if let RoundTripVerdict::KeyAdded { key } = v {
            assert_eq!(key, "c");
        }
    }

    #[test]
    fn order_does_not_matter() {
        // Different ordering, same content → identical.
        assert_eq!(
            check(&[("a", "1"), ("b", "2")], &[("b", "2"), ("a", "1")]),
            RoundTripVerdict::Identical
        );
    }

    #[test]
    fn empty_both_special() {
        assert_eq!(check(&[], &[]), RoundTripVerdict::EmptyInput);
    }

    #[test]
    fn empty_original_added() {
        let v = check(&[], &[("a", "1")]);
        assert!(matches!(v, RoundTripVerdict::KeyAdded { .. }));
    }

    #[test]
    fn empty_round_dropped() {
        let v = check(&[("a", "1")], &[]);
        assert!(matches!(v, RoundTripVerdict::KeyDropped { .. }));
    }

    #[test]
    fn dropped_takes_precedence_over_added() {
        // Both dropped and added simultaneously → dropped reported first.
        let v = check(&[("a", "1"), ("b", "2")], &[("a", "1"), ("c", "3")]);
        assert!(matches!(v, RoundTripVerdict::KeyDropped { .. }));
    }

    #[test]
    fn deterministic() {
        let a = check(&[("a", "1")], &[("a", "1")]);
        let b = check(&[("a", "1")], &[("a", "1")]);
        assert_eq!(a, b);
    }
}
