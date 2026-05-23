//! # WASM Data Segment Overlap Detect
//!
//! Detect overlapping linear-memory data segments. Each segment is
//! `(offset, length)`; two segments overlap if their byte ranges
//! intersect. Returns sorted overlapping pair indices.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §4.5.6 data instantiation; LLD wasm
//!  linker overlap-detection logic.
//!
//! Run with: cargo run --example wasm_data_segment_overlap
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OverlapVerdict {
    Ok {
        overlapping_pairs: Vec<(u32, u32)>,
        segment_count: u32,
    },
    InvalidConfig,
}

pub fn check(segments: &[(u32, u32)]) -> OverlapVerdict {
    if segments.is_empty() {
        return OverlapVerdict::InvalidConfig;
    }
    for (_, len) in segments {
        if *len == 0 {
            return OverlapVerdict::InvalidConfig;
        }
    }
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..segments.len() {
        for j in (i + 1)..segments.len() {
            let (o1, l1) = segments[i];
            let (o2, l2) = segments[j];
            let end1 = o1 + l1;
            let end2 = o2 + l2;
            if o1 < end2 && o2 < end1 {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    pairs.sort_unstable();
    OverlapVerdict::Ok {
        overlapping_pairs: pairs,
        segment_count: segments.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_data_segment_overlap")?;

    println!("disjoint: {:?}", check(&[(0, 10), (20, 10)]));
    println!("overlap: {:?}", check(&[(0, 20), (10, 10)]));
    println!("invalid: {:?}", check(&[(0, 0)]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn disjoint_no_overlap() {
        let v = check(&[(0, 10), (20, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert!(overlapping_pairs.is_empty());
        }
    }

    #[test]
    fn overlap_detected() {
        let v = check(&[(0, 20), (10, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert_eq!(overlapping_pairs, vec![(0, 1)]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), OverlapVerdict::InvalidConfig);
    }

    #[test]
    fn zero_length_rejected() {
        assert_eq!(check(&[(0, 0)]), OverlapVerdict::InvalidConfig);
    }

    #[test]
    fn touching_not_overlapping() {
        // [0,10) and [10,20) touch but don't overlap.
        let v = check(&[(0, 10), (10, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert!(overlapping_pairs.is_empty());
        }
    }

    #[test]
    fn nested_overlap_detected() {
        let v = check(&[(0, 100), (50, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert_eq!(overlapping_pairs, vec![(0, 1)]);
        }
    }

    #[test]
    fn segment_count_correct() {
        let v = check(&[(0, 10), (20, 10), (40, 10)]);
        if let OverlapVerdict::Ok { segment_count, .. } = v {
            assert_eq!(segment_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[(0, 10), (20, 10)]);
        let r2 = check(&[(0, 10), (20, 10)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn pairs_sorted() {
        let v = check(&[(0, 100), (50, 10), (60, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            for w in overlapping_pairs.windows(2) {
                assert!(w[0] <= w[1]);
            }
        }
    }

    #[test]
    fn many_segments_handled() {
        let segments: Vec<(u32, u32)> = (0..30).map(|i| (i * 100, 10)).collect();
        let v = check(&segments);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert!(overlapping_pairs.is_empty());
        }
    }

    #[test]
    fn single_segment_no_overlap() {
        let v = check(&[(0, 10)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert!(overlapping_pairs.is_empty());
        }
    }

    #[test]
    fn three_way_overlap_all_pairs() {
        let v = check(&[(0, 100), (50, 100), (75, 100)]);
        if let OverlapVerdict::Ok {
            overlapping_pairs, ..
        } = v
        {
            assert_eq!(overlapping_pairs.len(), 3);
        }
    }
}
