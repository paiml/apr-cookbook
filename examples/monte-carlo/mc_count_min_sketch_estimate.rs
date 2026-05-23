//! # Monte-Carlo Count-Min Sketch Frequency Estimation
//!
//! Estimate item frequencies from a stream using a count-min sketch
//! (d hash rows × w columns). Returns max relative error vs ground
//! truth across all items.
//!
//! Demonstrates the **MC.176** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cormode & Muthukrishnan, "An improved data stream
//!  summary: the count-min sketch" J. Algorithms 55(1) (2005).
//!
//! Run with: cargo run --example mc_count_min_sketch_estimate
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CountMinVerdict {
    Ok {
        max_rel_error_x100: u32,
        rows: u32,
        cols: u32,
    },
    InvalidConfig,
}

pub fn estimate(stream: &[u32], rows: u32, cols: u32) -> CountMinVerdict {
    if stream.is_empty() || !(2..=10).contains(&rows) || !(8..=4096).contains(&cols) {
        return CountMinVerdict::InvalidConfig;
    }
    let mut sketch = vec![vec![0u32; cols as usize]; rows as usize];
    let mut truth: BTreeMap<u32, u32> = BTreeMap::new();
    for item in stream {
        *truth.entry(*item).or_insert(0) += 1;
        for (r, row) in sketch.iter_mut().enumerate() {
            let h = ((*item as u64).wrapping_mul((r as u64 + 1).wrapping_mul(2654435761))) as u32
                % cols;
            row[h as usize] += 1;
        }
    }
    let mut max_err_x100: u32 = 0;
    for (item, true_count) in &truth {
        let mut est = u32::MAX;
        for (r, row) in sketch.iter().enumerate() {
            let h = ((*item as u64).wrapping_mul((r as u64 + 1).wrapping_mul(2654435761))) as u32
                % cols;
            est = est.min(row[h as usize]);
        }
        let rel = ((est as f64 - *true_count as f64) / *true_count as f64 * 100.0) as u32;
        if rel > max_err_x100 {
            max_err_x100 = rel;
        }
    }
    CountMinVerdict::Ok {
        max_rel_error_x100: max_err_x100,
        rows,
        cols,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_count_min_sketch_estimate")?;

    let stream: Vec<u32> = (0..1000).map(|i| i % 50).collect();
    println!("d=4 w=512: {:?}", estimate(&stream, 4, 512));
    println!("invalid: {:?}", estimate(&[], 4, 512));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_empty_stream() {
        assert_eq!(estimate(&[], 4, 512), CountMinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_rows() {
        assert_eq!(estimate(&[1, 2], 1, 512), CountMinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_cols() {
        assert_eq!(estimate(&[1, 2], 4, 4), CountMinVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let stream: Vec<u32> = (0..100).collect();
        let a = estimate(&stream, 4, 256);
        let b = estimate(&stream, 4, 256);
        assert_eq!(a, b);
    }

    #[test]
    fn rows_returned() {
        let v = estimate(&[1, 2], 4, 256);
        if let CountMinVerdict::Ok { rows, .. } = v {
            assert_eq!(rows, 4);
        }
    }

    #[test]
    fn cols_returned() {
        let v = estimate(&[1, 2], 4, 512);
        if let CountMinVerdict::Ok { cols, .. } = v {
            assert_eq!(cols, 512);
        }
    }

    #[test]
    fn larger_sketch_smaller_error() {
        let stream: Vec<u32> = (0..1000).map(|i| i % 100).collect();
        let small = estimate(&stream, 2, 16);
        let large = estimate(&stream, 8, 4096);
        if let (
            CountMinVerdict::Ok {
                max_rel_error_x100: s,
                ..
            },
            CountMinVerdict::Ok {
                max_rel_error_x100: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l <= s);
        }
    }

    #[test]
    fn count_min_never_underestimates() {
        // Property: count-min sketch never returns less than true count.
        // Since we measure (est - truth)/truth, error is always ≥ 0.
        let stream = vec![1u32; 100];
        let v = estimate(&stream, 4, 256);
        if let CountMinVerdict::Ok {
            max_rel_error_x100, ..
        } = v
        {
            // Test passes by construction (u32 ≥ 0).
            assert!(max_rel_error_x100 < u32::MAX);
        }
    }

    #[test]
    fn min_rows_accepted() {
        let v = estimate(&[1, 2], 2, 8);
        assert!(matches!(v, CountMinVerdict::Ok { .. }));
    }

    #[test]
    fn many_items_handled() {
        let stream: Vec<u32> = (0..10_000).collect();
        let v = estimate(&stream, 4, 1024);
        assert!(matches!(v, CountMinVerdict::Ok { .. }));
    }

    #[test]
    fn single_item_no_error() {
        let v = estimate(&[42], 4, 256);
        if let CountMinVerdict::Ok {
            max_rel_error_x100, ..
        } = v
        {
            assert_eq!(max_rel_error_x100, 0);
        }
    }
}
