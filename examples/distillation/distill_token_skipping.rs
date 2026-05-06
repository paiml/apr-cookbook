//! # Distillation Easy-Token Skipping
//!
//! Skip distillation loss on tokens where teacher entropy is below
//! threshold (teacher already very confident). Saves compute by
//! focusing on hard tokens.
//!
//! Demonstrates the **DIST.39** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Selective backprop / curriculum learning over difficulty.
//!
//! Run with: cargo run --example distill_token_skipping
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkipVerdict {
    Ok {
        skip_mask: Vec<bool>,
        kept_count: u32,
        skip_pct: f64,
    },
    EmptyEntropies,
    InvalidThreshold,
}

pub fn build_mask(token_entropies: &[f64], skip_threshold: f64) -> SkipVerdict {
    if token_entropies.is_empty() {
        return SkipVerdict::EmptyEntropies;
    }
    if !skip_threshold.is_finite() || skip_threshold < 0.0 {
        return SkipVerdict::InvalidThreshold;
    }
    if token_entropies.iter().any(|e| !e.is_finite() || *e < 0.0) {
        return SkipVerdict::InvalidThreshold;
    }
    let mut skip_mask = Vec::with_capacity(token_entropies.len());
    let mut kept_count = 0u32;
    for &e in token_entropies {
        let skip = e < skip_threshold;
        if !skip {
            kept_count += 1;
        }
        skip_mask.push(skip);
    }
    let total = token_entropies.len() as f64;
    let skipped = total - f64::from(kept_count);
    let skip_pct = (skipped / total) * 100.0;
    SkipVerdict::Ok {
        skip_mask,
        kept_count,
        skip_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_token_skipping")?;

    println!(
        "typical: {:?}",
        build_mask(&[0.1, 2.0, 0.05, 3.5, 0.3], 0.5)
    );
    println!("all hard: {:?}", build_mask(&[1.0, 2.0, 3.0], 0.5));
    println!("all easy: {:?}", build_mask(&[0.1, 0.2, 0.3], 0.5));
    println!("empty: {:?}", build_mask(&[], 0.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn easy_tokens_skipped() {
        let v = build_mask(&[0.1, 2.0, 0.05, 3.5, 0.3], 0.5);
        if let SkipVerdict::Ok { kept_count, .. } = v {
            // 2.0 and 3.5 are kept; 0.1, 0.05, 0.3 skipped.
            assert_eq!(kept_count, 2);
        }
    }

    #[test]
    fn all_hard_kept() {
        let v = build_mask(&[1.0, 2.0, 3.0], 0.5);
        if let SkipVerdict::Ok { kept_count, .. } = v {
            assert_eq!(kept_count, 3);
        }
    }

    #[test]
    fn all_easy_skipped() {
        let v = build_mask(&[0.1, 0.2, 0.3], 0.5);
        if let SkipVerdict::Ok { kept_count, .. } = v {
            assert_eq!(kept_count, 0);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(build_mask(&[], 0.5), SkipVerdict::EmptyEntropies);
    }

    #[test]
    fn negative_threshold_rejected() {
        assert_eq!(build_mask(&[1.0], -0.5), SkipVerdict::InvalidThreshold);
    }

    #[test]
    fn nan_entropy_rejected() {
        assert_eq!(build_mask(&[f64::NAN], 0.5), SkipVerdict::InvalidThreshold);
    }

    #[test]
    fn skip_pct_correct() {
        let v = build_mask(&[0.1, 0.2, 1.0, 2.0], 0.5);
        if let SkipVerdict::Ok { skip_pct, .. } = v {
            // 2 of 4 skipped → 50%.
            assert!((skip_pct - 50.0).abs() < 1e-6);
        }
    }

    #[test]
    fn skip_mask_length_matches_input() {
        let v = build_mask(&[0.1, 0.2, 1.0, 2.0], 0.5);
        if let SkipVerdict::Ok { skip_mask, .. } = v {
            assert_eq!(skip_mask.len(), 4);
        }
    }

    #[test]
    fn boundary_at_threshold_kept() {
        // At threshold = 0.5, e=0.5 → not skipped (only e < 0.5 is skipped).
        let v = build_mask(&[0.5], 0.5);
        if let SkipVerdict::Ok { kept_count, .. } = v {
            assert_eq!(kept_count, 1);
        }
    }

    #[test]
    fn zero_threshold_keeps_all() {
        let v = build_mask(&[0.0, 1.0, 2.0], 0.0);
        if let SkipVerdict::Ok { kept_count, .. } = v {
            assert_eq!(kept_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let a = build_mask(&[0.1, 1.0, 2.0], 0.5);
        let b = build_mask(&[0.1, 1.0, 2.0], 0.5);
        assert_eq!(a, b);
    }
}
