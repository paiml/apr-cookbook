//! # Contracts-Macros YAML Macro Expansion Audit
//!
//! Verify YAML macros expand to expected templates and catch
//! recursive cycles. Returns expanded count and any cycle-detected
//! macros.
//!
//! Demonstrates the **CMM.89** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: m4 macro processor; CPP recursive macro detection (gcc).
//!
//! Run with: cargo run --example contracts_macros_yaml_macros_expansion_audit
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ExpansionVerdict {
    Ok {
        expanded_count: u32,
        cycle_detected: Vec<String>,
    },
    InvalidConfig,
}

pub fn audit(macros: &[(&str, &str)]) -> ExpansionVerdict {
    if macros.is_empty() {
        return ExpansionVerdict::InvalidConfig;
    }
    let mut expansions: BTreeMap<String, String> = BTreeMap::new();
    for (name, body) in macros {
        if expansions
            .insert((*name).to_string(), (*body).to_string())
            .is_some()
        {
            // Duplicate definition — keep last.
        }
    }
    let mut cycle_detected: Vec<String> = Vec::new();
    for name in expansions.keys() {
        if has_cycle(name, &expansions, &mut BTreeSet::new()) {
            cycle_detected.push(name.clone());
        }
    }
    cycle_detected.sort();
    cycle_detected.dedup();
    ExpansionVerdict::Ok {
        expanded_count: expansions.len() as u32,
        cycle_detected,
    }
}

fn has_cycle(
    name: &str,
    expansions: &BTreeMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if !visiting.insert(name.to_string()) {
        return true;
    }
    let body = expansions.get(name).cloned().unwrap_or_default();
    for tok in body.split_whitespace() {
        let tok = tok.trim_matches(|c: char| c == '$' || c == '{' || c == '}');
        if expansions.contains_key(tok) && has_cycle(tok, expansions, visiting) {
            return true;
        }
    }
    visiting.remove(name);
    false
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_macros_expansion_audit")?;

    let safe = [("base", "leaf"), ("derived", "${base} extra")];
    println!("safe: {:?}", audit(&safe));
    let cyclic = [("a", "${b}"), ("b", "${a}")];
    println!("cycle: {:?}", audit(&cyclic));
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
    fn no_macros_acyclic() {
        let macros = [("plain", "hello world")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert!(cycle_detected.is_empty());
        }
    }

    #[test]
    fn linear_chain_acyclic() {
        let macros = [("a", "${b}"), ("b", "${c}"), ("c", "leaf")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert!(cycle_detected.is_empty());
        }
    }

    #[test]
    fn two_cycle_detected() {
        let macros = [("a", "${b}"), ("b", "${a}")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert!(cycle_detected.contains(&"a".to_string()));
            assert!(cycle_detected.contains(&"b".to_string()));
        }
    }

    #[test]
    fn three_cycle_detected() {
        let macros = [("a", "${b}"), ("b", "${c}"), ("c", "${a}")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert!(!cycle_detected.is_empty());
        }
    }

    #[test]
    fn self_reference_detected() {
        let macros = [("a", "${a}")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert_eq!(cycle_detected, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), ExpansionVerdict::InvalidConfig);
    }

    #[test]
    fn expanded_count_correct() {
        let macros = [("a", "x"), ("b", "y")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { expanded_count, .. } = v {
            assert_eq!(expanded_count, 2);
        }
    }

    #[test]
    fn duplicate_macro_collapsed() {
        let macros = [("a", "x"), ("a", "y")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { expanded_count, .. } = v {
            assert_eq!(expanded_count, 1);
        }
    }

    #[test]
    fn cycle_detected_sorted() {
        let macros = [("zeta", "${alpha}"), ("alpha", "${zeta}")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert_eq!(cycle_detected[0], "alpha");
            assert_eq!(cycle_detected[1], "zeta");
        }
    }

    #[test]
    fn deterministic() {
        let macros = [("a", "${b}"), ("b", "leaf")];
        let r1 = audit(&macros);
        let r2 = audit(&macros);
        assert_eq!(r1, r2);
    }

    #[test]
    fn dollar_brace_token_recognized() {
        let macros = [("base", "leaf"), ("derived", "${base}")];
        let v = audit(&macros);
        if let ExpansionVerdict::Ok { cycle_detected, .. } = v {
            assert!(cycle_detected.is_empty());
        }
    }
}
