//! # Contracts-Macros Obligation Namespace Collision
//!
//! Flag obligations whose short name (last segment) collides across
//! different namespaces. Returns colliding short-names with their
//! full namespace paths.
//!
//! Demonstrates the **CMM.112** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: namespace collision in module systems (Rust modules,
//!  Python imports); ambiguous-name diagnosis.
//!
//! Run with: cargo run --example contracts_macros_obligation_namespace_collision
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CollisionVerdict {
    Ok {
        collisions: BTreeMap<String, Vec<String>>,
    },
    InvalidConfig,
}

pub fn audit(qualified_names: &[&str]) -> CollisionVerdict {
    if qualified_names.is_empty() {
        return CollisionVerdict::InvalidConfig;
    }
    let mut by_short: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for qname in qualified_names {
        let short = qname.rsplit('.').next().unwrap_or(qname);
        by_short
            .entry(short.to_string())
            .or_default()
            .push((*qname).to_string());
    }
    let collisions: BTreeMap<String, Vec<String>> = by_short
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .collect();
    CollisionVerdict::Ok { collisions }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_namespace_collision")?;

    let names = ["a.foo", "b.foo", "a.bar", "c.bar"];
    println!("audit: {:?}", audit(&names));
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
    fn no_collision_empty() {
        let v = audit(&["a.foo", "b.bar"]);
        if let CollisionVerdict::Ok { collisions } = v {
            assert!(collisions.is_empty());
        }
    }

    #[test]
    fn collision_detected() {
        let v = audit(&["a.foo", "b.foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            assert!(collisions.contains_key("foo"));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn paths_collected_in_collision() {
        let v = audit(&["a.foo", "b.foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            let paths = collisions.get("foo").unwrap();
            assert_eq!(paths.len(), 2);
        }
    }

    #[test]
    fn no_dot_in_name_uses_full_name() {
        let v = audit(&["foo", "foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            assert!(collisions.contains_key("foo"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a.foo", "b.foo"]);
        let r2 = audit(&["a.foo", "b.foo"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn three_way_collision() {
        let v = audit(&["a.foo", "b.foo", "c.foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            let paths = collisions.get("foo").unwrap();
            assert_eq!(paths.len(), 3);
        }
    }

    #[test]
    fn deep_namespace_uses_last_segment() {
        let v = audit(&["a.b.c.foo", "x.y.foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            assert!(collisions.contains_key("foo"));
        }
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["a.Foo", "b.foo"]);
        if let CollisionVerdict::Ok { collisions } = v {
            // Different case → no collision.
            assert!(collisions.is_empty());
        }
    }

    #[test]
    fn collisions_sorted_by_short() {
        let v = audit(&["a.zeta", "b.zeta", "a.alpha", "b.alpha"]);
        if let CollisionVerdict::Ok { collisions } = v {
            let keys: Vec<&String> = collisions.keys().collect();
            assert_eq!(keys, vec!["alpha", "zeta"]);
        }
    }

    #[test]
    fn many_namespaces_handled() {
        let names: Vec<String> = (0..20).map(|i| format!("ns{i}.foo")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let v = audit(&refs);
        if let CollisionVerdict::Ok { collisions } = v {
            assert_eq!(collisions.get("foo").unwrap().len(), 20);
        }
    }
}
