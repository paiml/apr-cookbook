//! # apr experiment view — Per-Run Param Diff Renderer
//!
//! `apr experiment view` shows a side-by-side hyperparameter comparison
//! between selected runs. This recipe builds the diff renderer as a pure
//! function: same key in both → "(unchanged)" omitted; only the changed
//! parameters appear; missing keys flagged per-side.
//!
//! Demonstrates the **EXPERIMENT.6** recipe for PMAT-102 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXPERIMENT-003
//!
//! Run with: cargo run --example cli_experiment_param_diff_renderer
//!
//! Added by PMAT-102 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffEntry {
    Changed { from: String, to: String },
    OnlyInA(String),
    OnlyInB(String),
}

pub fn diff_params(
    a: &BTreeMap<String, String>,
    b: &BTreeMap<String, String>,
) -> BTreeMap<String, DiffEntry> {
    let mut out: BTreeMap<String, DiffEntry> = BTreeMap::new();
    for (k, av) in a {
        match b.get(k) {
            None => {
                out.insert(k.clone(), DiffEntry::OnlyInA(av.clone()));
            }
            Some(bv) if av != bv => {
                out.insert(
                    k.clone(),
                    DiffEntry::Changed {
                        from: av.clone(),
                        to: bv.clone(),
                    },
                );
            }
            _ => {}
        }
    }
    for (k, bv) in b {
        if !a.contains_key(k) {
            out.insert(k.clone(), DiffEntry::OnlyInB(bv.clone()));
        }
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_param_diff_renderer")?;

    let mut a: BTreeMap<String, String> = BTreeMap::new();
    a.insert("lr".into(), "1e-4".into());
    a.insert("batch_size".into(), "32".into());
    a.insert("dropout".into(), "0.1".into());
    a.insert("seed".into(), "42".into());

    let mut b: BTreeMap<String, String> = BTreeMap::new();
    b.insert("lr".into(), "5e-5".into()); // changed
    b.insert("batch_size".into(), "32".into()); // unchanged (omitted)
    b.insert("dropout".into(), "0.1".into()); // unchanged
    b.insert("warmup_steps".into(), "1000".into()); // OnlyInB
                                                    // seed missing in B → OnlyInA

    println!("=== Param diff (run A → run B) ===");
    for (k, e) in diff_params(&a, &b) {
        println!("  {k:>15}  {e:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn unchanged_keys_omitted() {
        let a = map(&[("lr", "1e-4"), ("seed", "42")]);
        let b = map(&[("lr", "1e-4"), ("seed", "42")]);
        assert!(diff_params(&a, &b).is_empty());
    }

    #[test]
    fn changed_key_emitted_with_from_and_to() {
        let a = map(&[("lr", "1e-4")]);
        let b = map(&[("lr", "5e-5")]);
        let d = diff_params(&a, &b);
        match d.get("lr") {
            Some(DiffEntry::Changed { from, to }) => {
                assert_eq!(from, "1e-4");
                assert_eq!(to, "5e-5");
            }
            other => panic!("expected Changed, got {other:?}"),
        }
    }

    #[test]
    fn key_only_in_a_flagged() {
        let a = map(&[("seed", "42")]);
        let b: BTreeMap<String, String> = BTreeMap::new();
        let d = diff_params(&a, &b);
        assert_eq!(d.get("seed"), Some(&DiffEntry::OnlyInA("42".into())));
    }

    #[test]
    fn key_only_in_b_flagged() {
        let a: BTreeMap<String, String> = BTreeMap::new();
        let b = map(&[("warmup", "1000")]);
        let d = diff_params(&a, &b);
        assert_eq!(d.get("warmup"), Some(&DiffEntry::OnlyInB("1000".into())));
    }

    #[test]
    fn diff_keys_sorted_by_btreemap() {
        // Output keys must be alphabetically sorted for deterministic CI logs.
        let a = map(&[("z", "1"), ("m", "2"), ("a", "3")]);
        let b = map(&[("z", "9"), ("m", "8"), ("a", "7")]);
        let d = diff_params(&a, &b);
        let keys: Vec<&String> = d.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn empty_inputs_yield_empty_diff() {
        let d = diff_params(&BTreeMap::new(), &BTreeMap::new());
        assert!(d.is_empty());
    }
}
