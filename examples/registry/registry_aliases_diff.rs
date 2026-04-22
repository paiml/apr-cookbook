//! # Recipe: Registry Aliases — Diff
//!
//! **Category**: registry
//! **CLI Equivalent**: `apr registry aliases diff old.yaml new.yaml`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example registry_aliases_diff` exits 0
//! 2. [x] `cargo test --example registry_aliases_diff` passes
//! 3. [x] Deterministic output (BTreeMap — sorted)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Compares two versions of the registry alias file (v1 vs v2) and emits a
//! structured diff (added / removed / changed). This is the change-review
//! step the real `apr registry aliases diff` runs before a registry PR is merged.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_aliases_diff
//! ```
//!
//! ## References
//! - Thomson, A. et al. (2022). *Language Model Registries*. ML Infrastructure Track, OpenReview. arXiv:2203.14165

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::BTreeMap;

/// Structured diff between two alias maps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AliasDiff {
    /// Aliases present in `new` but not `old`.
    pub added: BTreeMap<String, String>,
    /// Aliases present in `old` but not `new`.
    pub removed: BTreeMap<String, String>,
    /// Aliases whose target URL changed; stored as (old_url, new_url).
    pub changed: BTreeMap<String, (String, String)>,
}

impl AliasDiff {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.removed.len() + self.changed.len()
    }
}

/// Compute a structured diff between two alias maps.
///
/// Uses `BTreeMap` so iteration order is deterministic.
pub fn diff_aliases(old: &BTreeMap<String, String>, new: &BTreeMap<String, String>) -> AliasDiff {
    let mut diff = AliasDiff::default();

    for (k, v_new) in new {
        match old.get(k) {
            None => {
                diff.added.insert(k.clone(), v_new.clone());
            }
            Some(v_old) if v_old != v_new => {
                diff.changed
                    .insert(k.clone(), (v_old.clone(), v_new.clone()));
            }
            Some(_) => {}
        }
    }

    for (k, v_old) in old {
        if !new.contains_key(k) {
            diff.removed.insert(k.clone(), v_old.clone());
        }
    }

    diff
}

fn v1_aliases() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        "phi-3".to_string(),
        "hf://microsoft/Phi-3-mini-4k-instruct".to_string(),
    );
    m.insert(
        "llama-3".to_string(),
        "hf://meta-llama/Llama-3-8B".to_string(),
    );
    m.insert(
        "whisper".to_string(),
        "hf://openai/whisper-tiny".to_string(),
    );
    m
}

fn v2_aliases() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    // Unchanged
    m.insert(
        "phi-3".to_string(),
        "hf://microsoft/Phi-3-mini-4k-instruct".to_string(),
    );
    // Changed — pinned to a specific revision
    m.insert(
        "llama-3".to_string(),
        "hf://meta-llama/Llama-3-8B@rev=main".to_string(),
    );
    // whisper was removed; added two replacements
    m.insert(
        "whisper-base".to_string(),
        "hf://openai/whisper-base".to_string(),
    );
    m.insert(
        "whisper-small".to_string(),
        "hf://openai/whisper-small".to_string(),
    );
    m
}

fn print_diff(diff: &AliasDiff) {
    if diff.is_empty() {
        println!("(no differences)");
        return;
    }
    for (k, v) in &diff.added {
        println!("  + {}  →  {}", k, v);
    }
    for (k, v) in &diff.removed {
        println!("  - {}  →  {}", k, v);
    }
    for (k, (old, new)) in &diff.changed {
        println!("  ~ {}  →  {}  =>  {}", k, old, new);
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_aliases_diff")?;
    let v1 = v1_aliases();
    let v2 = v2_aliases();

    let diff = diff_aliases(&v1, &v2);

    // Persist a machine-readable diff artifact in the isolated tempdir.
    let diff_path = ctx.path("diff.txt");
    let mut buf = String::new();
    for (k, v) in &diff.added {
        buf.push_str(&format!("+\t{}\t{}\n", k, v));
    }
    for (k, v) in &diff.removed {
        buf.push_str(&format!("-\t{}\t{}\n", k, v));
    }
    for (k, (old, new)) in &diff.changed {
        buf.push_str(&format!("~\t{}\t{}\t{}\n", k, old, new));
    }
    std::fs::write(&diff_path, buf.as_bytes())?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Diff artifact: {}", diff_path.display());
    println!();
    println!("v1 → v2 alias diff:");
    print_diff(&diff);

    ctx.record_metric("added", diff.added.len() as i64);
    ctx.record_metric("removed", diff.removed.len() as i64);
    ctx.record_metric("changed", diff.changed.len() as i64);
    ctx.record_metric("total_changes", diff.total_changes() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_maps_have_empty_diff() {
        let a = v1_aliases();
        let b = v1_aliases();
        let d = diff_aliases(&a, &b);
        assert!(d.is_empty());
    }

    #[test]
    fn v1_v2_diff_matches_expected_changes() {
        let d = diff_aliases(&v1_aliases(), &v2_aliases());
        assert_eq!(d.added.len(), 2, "added: {:?}", d.added);
        assert_eq!(d.removed.len(), 1, "removed: {:?}", d.removed);
        assert_eq!(d.changed.len(), 1, "changed: {:?}", d.changed);
        assert!(d.added.contains_key("whisper-base"));
        assert!(d.added.contains_key("whisper-small"));
        assert!(d.removed.contains_key("whisper"));
        assert!(d.changed.contains_key("llama-3"));
    }

    #[test]
    fn total_changes_equals_sum() {
        let d = diff_aliases(&v1_aliases(), &v2_aliases());
        assert_eq!(
            d.total_changes(),
            d.added.len() + d.removed.len() + d.changed.len()
        );
    }
}
