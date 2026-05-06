//! # apr rosetta fingerprint — Diff Mode (`MODEL_B`)
//!
//! `apr rosetta fingerprint <MODEL_A> <MODEL_B>` enables diff mode: two
//! fingerprints are computed and the per-tensor stat-tuple
//! `(mean, std, min, max, l2_norm)` is compared with absolute tolerance.
//! This recipe builds the diff function and asserts the contract:
//! identical fingerprints diff to zero, missing tensors are reported
//! per-side, and the report is sorted for deterministic CI logs.
//!
//! Demonstrates the **ROSETTA-FINGERPRINT.3** recipe for PMAT-097 (fingerprint coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-201
//!
//! Run with: cargo run --example cli_rosetta_fingerprint_diff_mode
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct StatTuple {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffEntry {
    Match,
    Differ {
        max_field_drift: f64,
        worst_field: &'static str,
    },
    OnlyInA,
    OnlyInB,
}

pub fn diff_fingerprints(
    a: &BTreeMap<String, StatTuple>,
    b: &BTreeMap<String, StatTuple>,
    abs_tol: f64,
) -> BTreeMap<String, DiffEntry> {
    let mut out: BTreeMap<String, DiffEntry> = BTreeMap::new();
    for (name, av) in a {
        match b.get(name) {
            None => {
                out.insert(name.clone(), DiffEntry::OnlyInA);
            }
            Some(bv) => {
                let pairs = [
                    ("mean", av.mean - bv.mean),
                    ("std", av.std - bv.std),
                    ("min", av.min - bv.min),
                    ("max", av.max - bv.max),
                    ("l2_norm", av.l2_norm - bv.l2_norm),
                ];
                let (worst_field, worst_drift) =
                    pairs.iter().fold(("mean", 0.0_f64), |(name, max), (n, d)| {
                        if d.abs() > max.abs() {
                            (*n, *d)
                        } else {
                            (name, max)
                        }
                    });
                if worst_drift.abs() <= abs_tol {
                    out.insert(name.clone(), DiffEntry::Match);
                } else {
                    out.insert(
                        name.clone(),
                        DiffEntry::Differ {
                            max_field_drift: worst_drift.abs(),
                            worst_field,
                        },
                    );
                }
            }
        }
    }
    for name in b.keys() {
        if !a.contains_key(name) {
            out.insert(name.clone(), DiffEntry::OnlyInB);
        }
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_fingerprint_diff_mode")?;

    let mut a: BTreeMap<String, StatTuple> = BTreeMap::new();
    a.insert(
        "embed_tokens".into(),
        StatTuple {
            mean: 0.0,
            std: 0.02,
            min: -0.1,
            max: 0.1,
            l2_norm: 12.5,
        },
    );
    a.insert(
        "lm_head".into(),
        StatTuple {
            mean: 0.0,
            std: 0.04,
            min: -0.2,
            max: 0.2,
            l2_norm: 18.0,
        },
    );

    let mut b = a.clone();
    b.get_mut("lm_head").unwrap().std = 0.5; // big drift
    b.insert(
        "extra_tensor".into(),
        StatTuple {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            l2_norm: 0.0,
        },
    );

    println!("=== Recipe: cli_rosetta_fingerprint_diff_mode ===");
    for (name, entry) in diff_fingerprints(&a, &b, 0.01) {
        println!("  {name:>15}  {entry:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(mean: f64, std: f64) -> StatTuple {
        StatTuple {
            mean,
            std,
            min: mean - std,
            max: mean + std,
            l2_norm: std,
        }
    }

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_fingerprints_yield_match() {
        let mut a = BTreeMap::new();
        a.insert("x".into(), st(0.0, 1.0));
        let b = a.clone();
        let d = diff_fingerprints(&a, &b, 1e-9);
        assert_eq!(d.get("x"), Some(&DiffEntry::Match));
    }

    #[test]
    fn drift_above_tolerance_flagged() {
        let mut a = BTreeMap::new();
        a.insert("x".into(), st(0.0, 1.0));
        let mut b = BTreeMap::new();
        b.insert("x".into(), st(0.0, 5.0)); // std drift = 4
        let d = diff_fingerprints(&a, &b, 0.1);
        match d.get("x") {
            Some(DiffEntry::Differ { worst_field, .. }) => {
                assert!(["std", "min", "max", "l2_norm"].contains(worst_field))
            }
            other => panic!("expected Differ, got {other:?}"),
        }
    }

    #[test]
    fn tensor_only_in_a_flagged() {
        let mut a = BTreeMap::new();
        a.insert("x".into(), st(0.0, 1.0));
        let b: BTreeMap<String, StatTuple> = BTreeMap::new();
        let d = diff_fingerprints(&a, &b, 0.1);
        assert_eq!(d.get("x"), Some(&DiffEntry::OnlyInA));
    }

    #[test]
    fn tensor_only_in_b_flagged() {
        let a: BTreeMap<String, StatTuple> = BTreeMap::new();
        let mut b = BTreeMap::new();
        b.insert("y".into(), st(0.0, 1.0));
        let d = diff_fingerprints(&a, &b, 0.1);
        assert_eq!(d.get("y"), Some(&DiffEntry::OnlyInB));
    }

    #[test]
    fn diff_report_keys_sorted_via_btreemap() {
        let mut a = BTreeMap::new();
        a.insert("z".into(), st(0.0, 1.0));
        a.insert("a".into(), st(0.0, 1.0));
        a.insert("m".into(), st(0.0, 1.0));
        let b = a.clone();
        let d = diff_fingerprints(&a, &b, 0.1);
        let keys: Vec<&String> = d.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
