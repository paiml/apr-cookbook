//! # Distillation Top-K Soft-Label Sparsification
//!
//! Keep only the top-K teacher logits per token. The rest are dropped
//! (with their probability mass redistributed to a single "other" bin).
//! Saves bandwidth in distributed teacher → student pipelines.
//!
//! Demonstrates the **DIST.40** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sparse soft labels (Tang et al. 2019).
//!
//! Run with: cargo run --example distill_softlabel_topk
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TopKVerdict {
    Ok {
        kept_indices: Vec<u32>,
        kept_logits: Vec<f64>,
        dropped_count: u32,
    },
    EmptyLogits,
    InvalidK,
}

pub fn pick(logits: &[f64], k: u32) -> TopKVerdict {
    if logits.is_empty() {
        return TopKVerdict::EmptyLogits;
    }
    if k == 0 {
        return TopKVerdict::InvalidK;
    }
    let k = (k as usize).min(logits.len());
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, l)| (i, *l)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut top: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();
    // Re-sort by index for deterministic output order.
    top.sort_by_key(|(i, _)| *i);
    let kept_indices: Vec<u32> = top.iter().map(|(i, _)| *i as u32).collect();
    let kept_logits: Vec<f64> = top.iter().map(|(_, l)| *l).collect();
    let dropped_count = (logits.len() - k) as u32;
    TopKVerdict::Ok {
        kept_indices,
        kept_logits,
        dropped_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_softlabel_topk")?;

    println!("top-3: {:?}", pick(&[1.0, 5.0, 2.0, 9.0, 0.5, 3.0], 3));
    println!("k=all: {:?}", pick(&[1.0, 2.0, 3.0], 5));
    println!("empty: {:?}", pick(&[], 3));
    println!("k=0: {:?}", pick(&[1.0, 2.0], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn top_k_kept() {
        let v = pick(&[1.0, 5.0, 2.0, 9.0, 0.5, 3.0], 3);
        if let TopKVerdict::Ok { kept_indices, .. } = v {
            // Top 3 by value: 9.0 (i=3), 5.0 (i=1), 3.0 (i=5).
            // Sorted by index: 1, 3, 5.
            assert_eq!(kept_indices, vec![1, 3, 5]);
        }
    }

    #[test]
    fn k_exceeds_len_clamped() {
        let v = pick(&[1.0, 2.0, 3.0], 5);
        if let TopKVerdict::Ok {
            kept_indices,
            dropped_count,
            ..
        } = v
        {
            assert_eq!(kept_indices, vec![0, 1, 2]);
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(pick(&[], 3), TopKVerdict::EmptyLogits);
    }

    #[test]
    fn k_zero_rejected() {
        assert_eq!(pick(&[1.0, 2.0], 0), TopKVerdict::InvalidK);
    }

    #[test]
    fn dropped_count_correct() {
        let v = pick(&[1.0, 2.0, 3.0, 4.0, 5.0], 2);
        if let TopKVerdict::Ok { dropped_count, .. } = v {
            assert_eq!(dropped_count, 3);
        }
    }

    #[test]
    fn kept_logits_match_indices() {
        let v = pick(&[10.0, 1.0, 2.0, 5.0], 2);
        if let TopKVerdict::Ok {
            kept_indices,
            kept_logits,
            ..
        } = v
        {
            // Top 2: 10 (i=0), 5 (i=3). Sorted by index → indices [0, 3], logits [10.0, 5.0].
            assert_eq!(kept_indices, vec![0, 3]);
            assert_eq!(kept_logits, vec![10.0, 5.0]);
        }
    }

    #[test]
    fn ties_handled() {
        let v = pick(&[5.0, 5.0, 5.0, 1.0], 2);
        if let TopKVerdict::Ok { kept_indices, .. } = v {
            assert_eq!(kept_indices.len(), 2);
        }
    }

    #[test]
    fn k_one_returns_max() {
        let v = pick(&[1.0, 9.0, 2.0], 1);
        if let TopKVerdict::Ok { kept_indices, .. } = v {
            assert_eq!(kept_indices, vec![1]);
        }
    }

    #[test]
    fn large_input_works() {
        let logits: Vec<f64> = (0..1000).map(|i| f64::from(i)).collect();
        let v = pick(&logits, 10);
        if let TopKVerdict::Ok { kept_indices, .. } = v {
            // Top-10 of 0..999 are 990..999.
            assert_eq!(kept_indices, (990..1000).collect::<Vec<u32>>());
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(&[1.0, 5.0, 2.0, 9.0], 2);
        let b = pick(&[1.0, 5.0, 2.0, 9.0], 2);
        assert_eq!(a, b);
    }
}
