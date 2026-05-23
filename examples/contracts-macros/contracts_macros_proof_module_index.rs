//! # Contracts-Macros Proof Module Index
//!
//! Build an index over Lean proof modules by domain prefix
//! (e.g. `APR.Cookbook.Tui.*` → `tui`). Returns count per domain
//! and which modules went unclassified (no matching prefix).
//!
//! Demonstrates the **CMM.69** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mathlib4 namespace conventions; module hierarchy spec.
//!
//! Run with: cargo run --example contracts_macros_proof_module_index
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum IndexVerdict {
    Ok {
        per_domain: BTreeMap<String, u32>,
        unclassified: Vec<String>,
    },
    InvalidConfig,
}

pub fn index(modules: &[&str], domains: &[&str]) -> IndexVerdict {
    if modules.is_empty() || domains.is_empty() {
        return IndexVerdict::InvalidConfig;
    }
    let mut per_domain: BTreeMap<String, u32> = BTreeMap::new();
    for d in domains {
        per_domain.insert((*d).to_string(), 0);
    }
    let mut unclassified: Vec<String> = Vec::new();
    for m in modules {
        let mut matched = false;
        for d in domains {
            // Domain prefix match: ".d." within the path (case-insensitive)
            // or starts-with match for top-level.
            let needle = format!(".{}.", d.to_lowercase());
            let starts = format!("{}.", d.to_lowercase());
            let lower_m = m.to_lowercase();
            if lower_m.starts_with(&starts) || lower_m.contains(&needle) {
                *per_domain.entry((*d).to_string()).or_insert(0) += 1;
                matched = true;
                break;
            }
        }
        if !matched {
            unclassified.push((*m).to_string());
        }
    }
    IndexVerdict::Ok {
        per_domain,
        unclassified,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_proof_module_index")?;

    let modules = [
        "APR.Cookbook.Tui.BreadcrumbCollapse",
        "APR.Cookbook.MonteCarlo.LeastConn",
        "Standalone.Module.Foo",
    ];
    let domains = ["Tui", "MonteCarlo", "ContractsMacros"];
    println!("index: {:?}", index(&modules, &domains));
    println!("invalid: {:?}", index(&[], &domains));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn classifies_modules_by_prefix() {
        let modules = ["A.Tui.X", "A.MonteCarlo.Y"];
        let domains = ["Tui", "MonteCarlo"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.get("Tui"), Some(&1));
            assert_eq!(per_domain.get("MonteCarlo"), Some(&1));
        }
    }

    #[test]
    fn unmatched_module_unclassified() {
        let modules = ["Standalone.Module.Foo"];
        let domains = ["Tui"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { unclassified, .. } = v {
            assert_eq!(unclassified, vec!["Standalone.Module.Foo".to_string()]);
        }
    }

    #[test]
    fn empty_modules_rejected() {
        let domains = ["Tui"];
        assert_eq!(index(&[], &domains), IndexVerdict::InvalidConfig);
    }

    #[test]
    fn empty_domains_rejected() {
        let modules = ["A.B"];
        assert_eq!(index(&modules, &[]), IndexVerdict::InvalidConfig);
    }

    #[test]
    fn case_insensitive_match() {
        let modules = ["APR.tui.X", "apr.TUI.Y"];
        let domains = ["Tui"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.get("Tui"), Some(&2));
        }
    }

    #[test]
    fn module_starts_with_domain() {
        let modules = ["Tui.Foo"];
        let domains = ["Tui"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.get("Tui"), Some(&1));
        }
    }

    #[test]
    fn first_match_wins() {
        let modules = ["A.Tui.MonteCarlo.X"];
        let domains = ["Tui", "MonteCarlo"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.get("Tui"), Some(&1));
            assert_eq!(per_domain.get("MonteCarlo"), Some(&0));
        }
    }

    #[test]
    fn multiple_modules_per_domain() {
        let modules = ["A.Tui.X", "A.Tui.Y", "A.Tui.Z"];
        let domains = ["Tui"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.get("Tui"), Some(&3));
        }
    }

    #[test]
    fn deterministic() {
        let modules = ["A.Tui.X"];
        let domains = ["Tui"];
        let r1 = index(&modules, &domains);
        let r2 = index(&modules, &domains);
        assert_eq!(r1, r2);
    }

    #[test]
    fn all_domains_present_in_index() {
        let modules = ["A.Tui.X"];
        let domains = ["Tui", "MonteCarlo", "ContractsMacros"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok { per_domain, .. } = v {
            assert_eq!(per_domain.len(), 3);
        }
    }

    #[test]
    fn only_unclassified() {
        let modules = ["X.Y", "Y.Z"];
        let domains = ["Tui"];
        let v = index(&modules, &domains);
        if let IndexVerdict::Ok {
            unclassified,
            per_domain,
        } = v
        {
            assert_eq!(unclassified.len(), 2);
            assert_eq!(per_domain.get("Tui"), Some(&0));
        }
    }
}
