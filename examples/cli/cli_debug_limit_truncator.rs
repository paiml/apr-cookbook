//! # apr debug — `--limit <N>` Output Truncator
//!
//! `apr debug --limit <N> <FILE>` caps the number of output lines (default
//! 256). This recipe builds the truncator with an "elision marker" that
//! tells the operator how many lines were skipped — never silently drop.
//! Hard cap at u32::MAX/2 to avoid pathological allocations.
//!
//! Demonstrates the **DEBUG.6** recipe for PMAT-101 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEBUG-003 + classic head(1) elision convention
//!
//! Run with: cargo run --example cli_debug_limit_truncator
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TruncationVerdict {
    Full(Vec<String>), // emitted everything
    Truncated { kept: Vec<String>, dropped: usize },
    LimitTooLarge, // refused due to allocation risk
}

const HARD_CAP: usize = (u32::MAX / 2) as usize;

pub fn truncate(lines: &[String], limit: usize) -> TruncationVerdict {
    if limit > HARD_CAP {
        return TruncationVerdict::LimitTooLarge;
    }
    if lines.len() <= limit {
        return TruncationVerdict::Full(lines.to_vec());
    }
    let kept: Vec<String> = lines.iter().take(limit).cloned().collect();
    TruncationVerdict::Truncated {
        kept,
        dropped: lines.len() - limit,
    }
}

pub fn render_with_marker(verdict: &TruncationVerdict) -> Vec<String> {
    match verdict {
        TruncationVerdict::Full(v) => v.clone(),
        TruncationVerdict::Truncated { kept, dropped } => {
            let mut out = kept.clone();
            out.push(format!(
                "… ({dropped} more lines elided; rerun with --limit {})",
                kept.len() + dropped
            ));
            out
        }
        TruncationVerdict::LimitTooLarge => vec!["error: --limit too large".into()],
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_limit_truncator")?;

    let lines: Vec<String> = (0..1000).map(|i| format!("line-{i:04}")).collect();
    let v = truncate(&lines, 5);
    for l in render_with_marker(&v) {
        println!("  {l}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_truncation_when_under_limit() {
        let lines = vec!["a".to_string(), "b".into(), "c".into()];
        let v = truncate(&lines, 10);
        assert_eq!(v, TruncationVerdict::Full(lines.clone()));
    }

    #[test]
    fn truncation_records_dropped_count() {
        let lines: Vec<String> = (0..10).map(|i| format!("l{i}")).collect();
        let v = truncate(&lines, 3);
        if let TruncationVerdict::Truncated { kept, dropped } = v {
            assert_eq!(kept.len(), 3);
            assert_eq!(dropped, 7);
        } else {
            panic!("expected Truncated");
        }
    }

    #[test]
    fn truncation_preserves_first_n_lines() {
        let lines: Vec<String> = (0..10).map(|i| format!("l{i}")).collect();
        let v = truncate(&lines, 3);
        if let TruncationVerdict::Truncated { kept, .. } = v {
            assert_eq!(kept, vec!["l0".to_string(), "l1".into(), "l2".into()]);
        }
    }

    #[test]
    fn render_includes_elision_marker() {
        let lines: Vec<String> = (0..5).map(|i| format!("l{i}")).collect();
        let v = truncate(&lines, 2);
        let rendered = render_with_marker(&v);
        // 2 kept + 1 marker line = 3.
        assert_eq!(rendered.len(), 3);
        assert!(rendered.last().unwrap().contains("3 more"));
    }

    #[test]
    fn full_render_has_no_elision() {
        let lines = vec!["a".to_string(), "b".into()];
        let v = truncate(&lines, 5);
        let rendered = render_with_marker(&v);
        assert!(rendered.iter().all(|l| !l.contains("elided")));
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let v = truncate(&[], 100);
        assert_eq!(v, TruncationVerdict::Full(vec![]));
    }

    #[test]
    fn pathological_limit_rejected() {
        // limit > u32::MAX/2 → refuse rather than risk allocation.
        let lines = vec!["a".to_string()];
        let v = truncate(&lines, HARD_CAP + 1);
        assert_eq!(v, TruncationVerdict::LimitTooLarge);
    }

    #[test]
    fn limit_at_exact_cap_passes() {
        let lines = vec!["a".to_string()];
        let v = truncate(&lines, HARD_CAP);
        assert!(matches!(v, TruncationVerdict::Full(_)));
    }
}
