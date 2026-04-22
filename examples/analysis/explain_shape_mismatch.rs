//! # Recipe: Explain — Tensor Shape Mismatch with Suggestions
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr explain --shape "64,128" --expected "64,256"`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example explain_shape_mismatch` exits 0
//! 2. [x] `cargo test --example explain_shape_mismatch` passes
//! 3. [x] Deterministic output (pure)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr explain --shape` in-process (no shell-out)
//! 10. [x] Unit tests cover rank, size, transpose, broadcast suggestions
//!
//! ## Learning Objective
//! Given two shapes, classifies the mismatch (rank vs. axis size vs. potential
//! transpose) and emits actionable suggestions. This is the heart of the
//! educational side of `apr explain` — turning cryptic dimension errors into
//! concrete fixes.
//!
//! ## Run Command
//! ```bash
//! cargo run --example explain_shape_mismatch
//! ```
//!
//! ## References
//! - Ko, A.J. & Myers, B.A. (2008). *Debugging Reinvented*. ICSE. DOI: 10.1145/1368088.1368132

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MismatchKind {
    RankMismatch,
    AxisSizeMismatch(usize),
    PossibleTranspose,
    BroadcastCandidate,
    Match,
}

#[derive(Debug, Clone)]
struct Diagnosis {
    kind: MismatchKind,
    suggestions: Vec<String>,
}

fn diagnose(actual: &[usize], expected: &[usize]) -> Diagnosis {
    if actual == expected {
        return Diagnosis {
            kind: MismatchKind::Match,
            suggestions: vec!["shapes already match".into()],
        };
    }
    if actual.len() != expected.len() {
        let mut sug = Vec::new();
        sug.push(format!(
            "rank differs: actual rank={}, expected rank={}",
            actual.len(),
            expected.len()
        ));
        // If one is rank+1 of the other with a leading 1, suggest squeeze/unsqueeze.
        if actual.len() + 1 == expected.len() && expected.first() == Some(&1) {
            sug.push("try: unsqueeze(0) — add leading batch dim".into());
        }
        if expected.len() + 1 == actual.len() && actual.first() == Some(&1) {
            sug.push("try: squeeze(0) — remove leading dim of size 1".into());
        }
        return Diagnosis {
            kind: MismatchKind::RankMismatch,
            suggestions: sug,
        };
    }
    // Same rank: check if reversing matches (transpose candidate).
    let reversed: Vec<usize> = actual.iter().rev().copied().collect();
    if reversed == expected {
        return Diagnosis {
            kind: MismatchKind::PossibleTranspose,
            suggestions: vec![format!("try: transpose({:?}) -> {:?}", actual, expected)],
        };
    }
    // Broadcast candidate: every differing axis on one side is 1.
    let broadcast_ok = actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a == e || *a == 1 || *e == 1);
    if broadcast_ok {
        return Diagnosis {
            kind: MismatchKind::BroadcastCandidate,
            suggestions: vec!["shape may broadcast; ensure op supports broadcasting".into()],
        };
    }
    // Locate first differing axis.
    let axis = actual
        .iter()
        .zip(expected.iter())
        .position(|(a, e)| a != e)
        .unwrap_or(0);
    Diagnosis {
        kind: MismatchKind::AxisSizeMismatch(axis),
        suggestions: vec![format!(
            "axis {} size {} != expected {}; reshape/pad/slice",
            axis, actual[axis], expected[axis]
        )],
    }
}

fn kind_label(k: &MismatchKind) -> String {
    match k {
        MismatchKind::RankMismatch => "RANK_MISMATCH".into(),
        MismatchKind::AxisSizeMismatch(a) => format!("AXIS_SIZE_MISMATCH[axis={}]", a),
        MismatchKind::PossibleTranspose => "POSSIBLE_TRANSPOSE".into(),
        MismatchKind::BroadcastCandidate => "BROADCAST_CANDIDATE".into(),
        MismatchKind::Match => "MATCH".into(),
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("explain_shape_mismatch")?;
    println!("=== Recipe: {} ===", ctx.name());

    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![64, 128], vec![64, 256]),
        (vec![64, 128], vec![128, 64]),
        (vec![64, 128], vec![1, 64, 128]),
        (vec![1, 128], vec![64, 128]),
        (vec![64, 128], vec![64, 128]),
    ];

    let mut entries = Vec::new();
    println!("\n--- Diagnoses ---");
    for (a, e) in &cases {
        let d = diagnose(a, e);
        println!("actual={:?} expected={:?} => {}", a, e, kind_label(&d.kind));
        for s in &d.suggestions {
            println!("    suggest: {}", s);
        }
        entries.push((a.clone(), e.clone(), d));
    }

    let report = json!({
        "recipe": ctx.name(),
        "cases": entries.iter().map(|(a, e, d)| json!({
            "actual": a,
            "expected": e,
            "kind": kind_label(&d.kind),
            "suggestions": d.suggestions,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("shape-mismatch.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_shapes() {
        let d = diagnose(&[2, 3], &[2, 3]);
        assert_eq!(d.kind, MismatchKind::Match);
    }

    #[test]
    fn rank_mismatch_detected() {
        let d = diagnose(&[2, 3], &[1, 2, 3]);
        assert_eq!(d.kind, MismatchKind::RankMismatch);
        assert!(d.suggestions.iter().any(|s| s.contains("unsqueeze")));
    }

    #[test]
    fn transpose_candidate() {
        let d = diagnose(&[64, 128], &[128, 64]);
        assert_eq!(d.kind, MismatchKind::PossibleTranspose);
    }

    #[test]
    fn axis_size_mismatch_identifies_axis() {
        let d = diagnose(&[64, 128], &[64, 256]);
        assert_eq!(d.kind, MismatchKind::AxisSizeMismatch(1));
    }

    #[test]
    fn broadcast_candidate_detected() {
        let d = diagnose(&[1, 128], &[64, 128]);
        assert_eq!(d.kind, MismatchKind::BroadcastCandidate);
    }
}
