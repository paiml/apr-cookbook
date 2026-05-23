//! # apr tensors — `--limit <N>` Truncator
//!
//! `apr tensors --limit <N>` caps the number of tensors shown.
//! `--limit 0` means unlimited (the default). This recipe builds the
//! truncator and asserts the contract: 0 = unlimited, N preserves the
//! first N tensors after filtering, output marks "K more elided"
//! when truncation happens.
//!
//! Demonstrates the **TENSORS.10** recipe for PMAT-110 (apr tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TENSORS-002
//!
//! Run with: cargo run --example cli_tensors_limit_truncator
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TruncationVerdict {
    Full,
    Truncated { kept: usize, elided: usize },
}

pub fn truncate(tensor_count: usize, limit: u32) -> TruncationVerdict {
    if limit == 0 {
        return TruncationVerdict::Full;
    }
    let l = limit as usize;
    if tensor_count <= l {
        TruncationVerdict::Full
    } else {
        TruncationVerdict::Truncated {
            kept: l,
            elided: tensor_count - l,
        }
    }
}

pub fn render_marker(v: &TruncationVerdict) -> Option<String> {
    if let TruncationVerdict::Truncated { kept, elided } = v {
        Some(format!(
            "… {elided} more (rerun with --limit {})",
            kept + elided
        ))
    } else {
        None
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tensors_limit_truncator")?;

    for (count, limit) in [(100, 0u32), (100, 200), (100, 10), (5, 100)] {
        let v = truncate(count, limit);
        let marker = render_marker(&v).unwrap_or_default();
        println!("count={count:>4} limit={limit:>4}  →  {v:?}  {marker}");
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
    fn limit_zero_is_unlimited() {
        // Default behavior: --limit 0 means unlimited.
        assert_eq!(truncate(1000, 0), TruncationVerdict::Full);
    }

    #[test]
    fn count_below_limit_is_full() {
        assert_eq!(truncate(5, 100), TruncationVerdict::Full);
    }

    #[test]
    fn count_above_limit_truncates() {
        let v = truncate(100, 10);
        assert_eq!(
            v,
            TruncationVerdict::Truncated {
                kept: 10,
                elided: 90
            }
        );
    }

    #[test]
    fn count_equal_to_limit_is_full() {
        // Boundary: count == limit means we kept everything.
        assert_eq!(truncate(50, 50), TruncationVerdict::Full);
    }

    #[test]
    fn marker_only_for_truncated() {
        assert!(render_marker(&TruncationVerdict::Full).is_none());
        let m = render_marker(&TruncationVerdict::Truncated {
            kept: 5,
            elided: 95,
        });
        assert!(m.is_some());
        assert!(m.unwrap().contains("95 more"));
    }

    #[test]
    fn marker_includes_full_count_for_rerun_hint() {
        let m = render_marker(&TruncationVerdict::Truncated {
            kept: 5,
            elided: 95,
        })
        .unwrap();
        // Hint should suggest --limit 100 (5 + 95) to see all.
        assert!(m.contains("--limit 100"));
    }

    #[test]
    fn empty_count_is_full() {
        assert_eq!(truncate(0, 100), TruncationVerdict::Full);
        assert_eq!(truncate(0, 0), TruncationVerdict::Full);
    }
}
