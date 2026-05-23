//! # Contracts-Macros Invariant Proof Language
//!
//! Validate proof-language declarations across invariants: must use
//! one of `lean`, `coq`, `agda`, `kani`, `manual`. Returns sorted
//! offending invariants and per-language counts.
//!
//! Demonstrates the **CMM.195** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA proof-attestation languages; in-toto layout
//!  proof-system enums.
//!
//! Run with: cargo run --example contracts_macros_invariant_proof_lang
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum ProofLangVerdict {
    Ok {
        offending_invariants: Vec<String>,
        per_language: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

const ALLOWED: &[&str] = &["lean", "coq", "agda", "kani", "manual"];

pub fn check(items: &[(&str, &str)]) -> ProofLangVerdict {
    if items.is_empty() {
        return ProofLangVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, u32> = ALLOWED.iter().map(|l| ((*l).to_string(), 0)).collect();
    for (id, lang) in items {
        if ALLOWED.contains(lang) {
            if let Some(c) = counts.get_mut(*lang) {
                *c += 1;
            }
        } else {
            offenders.push((*id).to_string());
        }
    }
    offenders.sort();
    ProofLangVerdict::Ok {
        offending_invariants: offenders,
        per_language: counts,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_proof_lang")?;

    let items = [("inv1", "lean"), ("inv2", "rust"), ("inv3", "coq")];
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
    fn allowed_language_no_offender() {
        let v = check(&[("a", "lean")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert!(offending_invariants.is_empty());
        }
    }

    #[test]
    fn unknown_language_offender() {
        let v = check(&[("a", "rust")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert_eq!(offending_invariants, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), ProofLangVerdict::InvalidConfig);
    }

    #[test]
    fn per_language_count_correct() {
        let v = check(&[("a", "lean"), ("b", "lean"), ("c", "coq")]);
        if let ProofLangVerdict::Ok { per_language, .. } = v {
            assert_eq!(per_language.get("lean"), Some(&2));
            assert_eq!(per_language.get("coq"), Some(&1));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", "lean")]);
        let r2 = check(&[("a", "lean")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", "rust"), ("alpha", "rust")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert_eq!(
                offending_invariants,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn manual_accepted() {
        let v = check(&[("a", "manual")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert!(offending_invariants.is_empty());
        }
    }

    #[test]
    fn case_sensitive() {
        // "Lean" (capitalized) is not in the lowercase ALLOWED list.
        let v = check(&[("a", "Lean")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert_eq!(offending_invariants, vec!["a".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("a", "lean")).collect();
        let v = check(&items);
        if let ProofLangVerdict::Ok { per_language, .. } = v {
            assert_eq!(per_language.get("lean"), Some(&30));
        }
    }

    #[test]
    fn all_languages_initialized_to_zero() {
        let v = check(&[("a", "lean")]);
        if let ProofLangVerdict::Ok { per_language, .. } = v {
            assert!(per_language.contains_key("coq"));
            assert!(per_language.contains_key("agda"));
            assert!(per_language.contains_key("kani"));
            assert!(per_language.contains_key("manual"));
        }
    }

    #[test]
    fn agda_kani_accepted() {
        let v = check(&[("a", "agda"), ("b", "kani")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert!(offending_invariants.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", "rust")]);
        if let ProofLangVerdict::Ok {
            offending_invariants,
            ..
        } = v
        {
            assert_eq!(offending_invariants, vec!["café".to_string()]);
        }
    }
}
