//! # Recipe: Probar Regression Diff Report
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr probar diff --baseline before.json --current after.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example probar_regression_diff` exits 0
//! 2. [x] `cargo test --example probar_regression_diff` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr probar diff` in-process (no shell-out)
//! 10. [x] Unit tests cover fixed, broken, still-broken, still-passing
//!
//! ## Learning Objective
//! Demonstrates a probar regression-diff report: takes two run snapshots and
//! classifies each property into Fixed, Broken, StillBroken, or StillPassing.
//! This is exactly the diff format `apr probar diff` emits when gated in CI.
//!
//! ## Run Command
//! ```bash
//! cargo run --example probar_regression_diff
//! ```
//!
//! ## References
//! - Claessen, K. & Hughes, J. (2000). *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs*. ICFP. DOI: 10.1145/351240.351266

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropStatus {
    Pass,
    Fail,
}

impl PropStatus {
    fn label(self) -> &'static str {
        match self {
            PropStatus::Pass => "pass",
            PropStatus::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    Fixed,        // fail -> pass
    Broken,       // pass -> fail
    StillBroken,  // fail -> fail
    StillPassing, // pass -> pass
}

impl Transition {
    pub fn label(&self) -> &'static str {
        match self {
            Transition::Fixed => "fixed",
            Transition::Broken => "broken",
            Transition::StillBroken => "still-broken",
            Transition::StillPassing => "still-passing",
        }
    }
}

pub fn classify(before: PropStatus, after: PropStatus) -> Transition {
    match (before, after) {
        (PropStatus::Fail, PropStatus::Pass) => Transition::Fixed,
        (PropStatus::Pass, PropStatus::Fail) => Transition::Broken,
        (PropStatus::Fail, PropStatus::Fail) => Transition::StillBroken,
        (PropStatus::Pass, PropStatus::Pass) => Transition::StillPassing,
    }
}

pub fn diff_snapshots(
    baseline: &BTreeMap<String, PropStatus>,
    current: &BTreeMap<String, PropStatus>,
) -> BTreeMap<String, Transition> {
    let names: std::collections::BTreeSet<_> =
        baseline.keys().chain(current.keys()).cloned().collect();
    let mut out = BTreeMap::new();
    for n in names {
        // Missing entries count as Pass (property didn't exist → no regression).
        let b = baseline.get(&n).copied().unwrap_or(PropStatus::Pass);
        let c = current.get(&n).copied().unwrap_or(PropStatus::Pass);
        out.insert(n, classify(b, c));
    }
    out
}

fn build_baseline() -> BTreeMap<String, PropStatus> {
    let mut m = BTreeMap::new();
    m.insert("abs_nonneg".into(), PropStatus::Pass);
    m.insert("reverse_reverse_id".into(), PropStatus::Pass);
    m.insert("division_non_zero".into(), PropStatus::Fail);
    m.insert("sort_idempotent".into(), PropStatus::Pass);
    m
}

fn build_current() -> BTreeMap<String, PropStatus> {
    let mut m = BTreeMap::new();
    m.insert("abs_nonneg".into(), PropStatus::Pass);
    m.insert("reverse_reverse_id".into(), PropStatus::Fail);
    m.insert("division_non_zero".into(), PropStatus::Pass);
    m.insert("sort_idempotent".into(), PropStatus::Pass);
    m.insert("new_property".into(), PropStatus::Fail);
    m
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("probar_regression_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let baseline = build_baseline();
    let current = build_current();
    let diff = diff_snapshots(&baseline, &current);

    let mut fixed = 0usize;
    let mut broken = 0usize;
    let mut still_broken = 0usize;
    let mut still_passing = 0usize;
    for (name, t) in &diff {
        println!("  {:<25} {}", name, t.label());
        match t {
            Transition::Fixed => fixed += 1,
            Transition::Broken => broken += 1,
            Transition::StillBroken => still_broken += 1,
            Transition::StillPassing => still_passing += 1,
        }
    }

    println!(
        "fixed={} broken={} still_broken={} still_passing={}",
        fixed, broken, still_broken, still_passing
    );

    let report = json!({
        "recipe": ctx.name(),
        "fixed": fixed,
        "broken": broken,
        "still_broken": still_broken,
        "still_passing": still_passing,
        "diff": diff.iter().map(|(n, t)| (n.clone(), t.label())).collect::<BTreeMap<_, _>>(),
        "baseline": baseline.iter().map(|(n, s)| (n.clone(), s.label())).collect::<BTreeMap<_, _>>(),
    });
    let path = ctx.path("probar-regression-diff.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("broken", broken as i64);
    ctx.record_metric("fixed", fixed as i64);
    ctx.record_metric("still_broken", still_broken as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_transition() {
        assert_eq!(
            classify(PropStatus::Fail, PropStatus::Pass),
            Transition::Fixed
        );
    }

    #[test]
    fn broken_transition() {
        assert_eq!(
            classify(PropStatus::Pass, PropStatus::Fail),
            Transition::Broken
        );
    }

    #[test]
    fn still_broken_transition() {
        assert_eq!(
            classify(PropStatus::Fail, PropStatus::Fail),
            Transition::StillBroken
        );
    }

    #[test]
    fn still_passing_transition() {
        assert_eq!(
            classify(PropStatus::Pass, PropStatus::Pass),
            Transition::StillPassing
        );
    }

    #[test]
    fn diff_handles_new_property() {
        let mut base = BTreeMap::new();
        base.insert("a".into(), PropStatus::Pass);
        let mut cur = BTreeMap::new();
        cur.insert("a".into(), PropStatus::Pass);
        cur.insert("new_b".into(), PropStatus::Fail);
        let d = diff_snapshots(&base, &cur);
        assert_eq!(d.get("new_b"), Some(&Transition::Broken));
    }
}
