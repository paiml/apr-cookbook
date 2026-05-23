//! # Contracts-Macros Obligation Lock File
//!
//! Generate a deterministic lockfile of obligation states. Same input
//! → same lockfile bytes (sorted keys, consistent newlines).
//!
//! Demonstrates the **CMM.62** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo's Cargo.lock determinism contract.
//!
//! Run with: cargo run --example contracts_macros_obligation_lock_file
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum LockVerdict {
    Ok { lock_file: String, entry_count: u32 },
    EmptyContract,
}

pub fn generate(obligations: &[(&str, &str, u32)]) -> LockVerdict {
    if obligations.is_empty() {
        return LockVerdict::EmptyContract;
    }
    let mut sorted: BTreeMap<&str, (&str, u32)> = BTreeMap::new();
    for (id, status, version) in obligations {
        sorted.insert(*id, (*status, *version));
    }
    let mut out = String::new();
    out.push_str("# Auto-generated lock file (do not edit).\n");
    for (id, (status, version)) in &sorted {
        out.push_str(&format!(
            "{id} = {{ status = \"{status}\", version = {version} }}\n"
        ));
    }
    LockVerdict::Ok {
        lock_file: out,
        entry_count: sorted.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_lock_file")?;

    let obligations = [("oblig_a", "proved", 1u32), ("oblig_b", "wip", 2)];
    println!("typical: {:?}", generate(&obligations));
    println!("empty: {:?}", generate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn order_invariant() {
        let unsorted = [("z", "wip", 1u32), ("a", "proved", 2)];
        let sorted = [("a", "proved", 2u32), ("z", "wip", 1)];
        assert_eq!(generate(&unsorted), generate(&sorted));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(generate(&[]), LockVerdict::EmptyContract);
    }

    #[test]
    fn entry_count_correct() {
        let v = generate(&[("a", "proved", 1u32), ("b", "wip", 2)]);
        if let LockVerdict::Ok { entry_count, .. } = v {
            assert_eq!(entry_count, 2);
        }
    }

    #[test]
    fn header_present() {
        let v = generate(&[("a", "proved", 1u32)]);
        if let LockVerdict::Ok { lock_file, .. } = v {
            assert!(lock_file.starts_with("#"));
        }
    }

    #[test]
    fn id_in_output() {
        let v = generate(&[("my_oblig", "proved", 1u32)]);
        if let LockVerdict::Ok { lock_file, .. } = v {
            assert!(lock_file.contains("my_oblig"));
        }
    }

    #[test]
    fn status_in_output() {
        let v = generate(&[("a", "wip", 1u32)]);
        if let LockVerdict::Ok { lock_file, .. } = v {
            assert!(lock_file.contains("\"wip\""));
        }
    }

    #[test]
    fn version_in_output() {
        let v = generate(&[("a", "proved", 42u32)]);
        if let LockVerdict::Ok { lock_file, .. } = v {
            assert!(lock_file.contains("version = 42"));
        }
    }

    #[test]
    fn duplicate_keys_collapse() {
        // BTreeMap insert dedups; later wins.
        let v = generate(&[("a", "wip", 1u32), ("a", "proved", 2)]);
        if let LockVerdict::Ok {
            entry_count,
            lock_file,
        } = v
        {
            assert_eq!(entry_count, 1);
            assert!(lock_file.contains("\"proved\""));
        }
    }

    #[test]
    fn many_entries() {
        let entries: Vec<(&str, &str, u32)> = (0..50).map(|_| ("o", "proved", 1u32)).collect();
        let v = generate(&entries);
        if let LockVerdict::Ok { entry_count, .. } = v {
            // All collapse to 1 since same key.
            assert_eq!(entry_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let o = [("a", "proved", 1u32)];
        let a = generate(&o);
        let b = generate(&o);
        assert_eq!(a, b);
    }
}
