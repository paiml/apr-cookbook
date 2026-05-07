//! # Contracts-Macros YAML Anchor Namespace
//!
//! Validate anchor namespacing: each anchor must use a documented
//! prefix (e.g., `&pkg.foo`, `&env.bar`). Returns sorted offending
//! anchors and the per-namespace count.
//!
//! Demonstrates the **CMM.169** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitLab CI/CD anchor-naming conventions; YAML 1.2 §6.9.1
//!  anchor production.
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_namespace
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum AnchorNsVerdict {
    Ok {
        offending_anchors: Vec<String>,
        count_per_ns: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn check(anchors: &[&str], allowed_prefixes: &[&str]) -> AnchorNsVerdict {
    if anchors.is_empty() || allowed_prefixes.is_empty() {
        return AnchorNsVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for prefix in allowed_prefixes {
        counts.insert((*prefix).to_string(), 0);
    }
    for anchor in anchors {
        let mut matched: Option<&str> = None;
        for prefix in allowed_prefixes {
            if anchor.starts_with(&format!("{prefix}.")) {
                matched = Some(prefix);
                break;
            }
        }
        match matched {
            Some(p) => {
                if let Some(c) = counts.get_mut(p) {
                    *c += 1;
                }
            }
            None => offenders.push((*anchor).to_string()),
        }
    }
    offenders.sort();
    offenders.dedup();
    AnchorNsVerdict::Ok {
        offending_anchors: offenders,
        count_per_ns: counts,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_namespace")?;

    let anchors = ["pkg.serde", "env.api_url", "raw_anchor"];
    println!("check: {:?}", check(&anchors, &["pkg", "env"]));
    println!("invalid: {:?}", check(&[], &["pkg"]));
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
    fn valid_namespaced_anchor() {
        let v = check(&["pkg.foo"], &["pkg"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }

    #[test]
    fn unnamespaced_anchor_offender() {
        let v = check(&["foo"], &["pkg"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert_eq!(offending_anchors, vec!["foo".to_string()]);
        }
    }

    #[test]
    fn empty_anchors_rejected() {
        assert_eq!(check(&[], &["pkg"]), AnchorNsVerdict::InvalidConfig);
    }

    #[test]
    fn empty_prefixes_rejected() {
        assert_eq!(check(&["foo"], &[]), AnchorNsVerdict::InvalidConfig);
    }

    #[test]
    fn count_per_ns_correct() {
        let v = check(&["pkg.a", "pkg.b", "env.c"], &["pkg", "env"]);
        if let AnchorNsVerdict::Ok { count_per_ns, .. } = v {
            assert_eq!(count_per_ns.get("pkg"), Some(&2));
            assert_eq!(count_per_ns.get("env"), Some(&1));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["pkg.a"], &["pkg"]);
        let r2 = check(&["pkg.a"], &["pkg"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted_and_deduped() {
        let v = check(&["zeta", "alpha", "zeta"], &["pkg"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert_eq!(
                offending_anchors,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn prefix_must_have_dot_separator() {
        // "pkgfoo" doesn't match prefix "pkg" without "." — would be flagged.
        let v = check(&["pkgfoo"], &["pkg"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert_eq!(offending_anchors, vec!["pkgfoo".to_string()]);
        }
    }

    #[test]
    fn many_anchors_handled() {
        let anchors: Vec<&str> = (0..30).map(|_| "pkg.x").collect();
        let v = check(&anchors, &["pkg"]);
        if let AnchorNsVerdict::Ok { count_per_ns, .. } = v {
            assert_eq!(count_per_ns.get("pkg"), Some(&30));
        }
    }

    #[test]
    fn unicode_anchor_supported() {
        let v = check(&["pkg.café"], &["pkg"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }

    #[test]
    fn multiple_namespaces_independent() {
        let v = check(&["a.x", "b.y"], &["a", "b"]);
        if let AnchorNsVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }
}
