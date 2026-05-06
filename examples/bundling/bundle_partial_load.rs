//! # Bundle Partial Load Index
//!
//! Load only requested tensors. Picker: given (toc, requested_names),
//! returns minimal byte-range fetch list with sequential merging
//! (adjacent ranges combined to one read).
//!
//! Demonstrates the **BUNDLE.24** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SafeTensors lazy-loading + HuggingFace partial weights.
//!
//! Run with: cargo run --example bundle_partial_load
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct TocEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, PartialEq)]
pub enum LoadVerdict {
    Ok {
        ranges: Vec<ByteRange>,
        total_bytes: u64,
    },
    EmptyRequest,
    UnknownTensor {
        name: String,
    },
}

pub fn plan(toc: &[TocEntry], requested: &[&str]) -> LoadVerdict {
    if requested.is_empty() {
        return LoadVerdict::EmptyRequest;
    }
    let by_name: BTreeMap<&str, &TocEntry> = toc.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut entries: Vec<&TocEntry> = Vec::new();
    for name in requested {
        match by_name.get(name) {
            Some(e) => entries.push(e),
            None => {
                return LoadVerdict::UnknownTensor {
                    name: (*name).to_string(),
                }
            }
        }
    }
    entries.sort_by_key(|e| e.offset);
    // Merge adjacent ranges.
    let mut ranges: Vec<ByteRange> = Vec::new();
    for e in entries {
        let new_range = ByteRange {
            start: e.offset,
            end: e.offset + e.size,
        };
        match ranges.last_mut() {
            Some(last) if last.end >= new_range.start => last.end = last.end.max(new_range.end),
            _ => ranges.push(new_range),
        }
    }
    let total_bytes: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    LoadVerdict::Ok {
        ranges,
        total_bytes,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_partial_load")?;

    let toc = [
        TocEntry {
            name: "a".to_string(),
            offset: 0,
            size: 1000,
        },
        TocEntry {
            name: "b".to_string(),
            offset: 1000,
            size: 500,
        },
        TocEntry {
            name: "c".to_string(),
            offset: 2000,
            size: 800,
        },
    ];

    println!("a + b (adjacent): {:?}", plan(&toc, &["a", "b"]));
    println!("a + c (gap): {:?}", plan(&toc, &["a", "c"]));
    println!("only a: {:?}", plan(&toc, &["a"]));
    println!("missing: {:?}", plan(&toc, &["missing"]));
    println!("empty: {:?}", plan(&toc, &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_toc() -> Vec<TocEntry> {
        vec![
            TocEntry {
                name: "a".to_string(),
                offset: 0,
                size: 1000,
            },
            TocEntry {
                name: "b".to_string(),
                offset: 1000,
                size: 500,
            },
            TocEntry {
                name: "c".to_string(),
                offset: 2000,
                size: 800,
            },
        ]
    }

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn adjacent_tensors_merged() {
        let v = plan(&typical_toc(), &["a", "b"]);
        if let LoadVerdict::Ok { ranges, .. } = v {
            assert_eq!(ranges.len(), 1);
            assert_eq!(ranges[0].start, 0);
            assert_eq!(ranges[0].end, 1500);
        }
    }

    #[test]
    fn gap_tensors_separate_ranges() {
        let v = plan(&typical_toc(), &["a", "c"]);
        if let LoadVerdict::Ok { ranges, .. } = v {
            assert_eq!(ranges.len(), 2);
        }
    }

    #[test]
    fn single_tensor_one_range() {
        let v = plan(&typical_toc(), &["a"]);
        if let LoadVerdict::Ok { ranges, .. } = v {
            assert_eq!(ranges.len(), 1);
        }
    }

    #[test]
    fn empty_request_rejected() {
        assert_eq!(plan(&typical_toc(), &[]), LoadVerdict::EmptyRequest);
    }

    #[test]
    fn unknown_tensor_rejected() {
        let v = plan(&typical_toc(), &["missing"]);
        assert!(matches!(v, LoadVerdict::UnknownTensor { .. }));
    }

    #[test]
    fn total_bytes_sum_correct() {
        let v = plan(&typical_toc(), &["a", "c"]);
        if let LoadVerdict::Ok { total_bytes, .. } = v {
            assert_eq!(total_bytes, 1800);
        }
    }

    #[test]
    fn out_of_order_request_sorted() {
        let v = plan(&typical_toc(), &["c", "a"]);
        if let LoadVerdict::Ok { ranges, .. } = v {
            // Ranges should be in offset order.
            assert!(ranges[0].start < ranges[1].start);
        }
    }

    #[test]
    fn all_three_two_ranges_after_merge() {
        // a (0..1000) + b (1000..1500) merge → [0..1500]; c separate.
        let v = plan(&typical_toc(), &["a", "b", "c"]);
        if let LoadVerdict::Ok { ranges, .. } = v {
            assert_eq!(ranges.len(), 2);
        }
    }

    #[test]
    fn empty_toc_unknown_tensor() {
        let v = plan(&[], &["a"]);
        assert!(matches!(v, LoadVerdict::UnknownTensor { .. }));
    }

    #[test]
    fn duplicate_request_handled() {
        // Same tensor requested twice → fetched once.
        let v = plan(&typical_toc(), &["a", "a"]);
        if let LoadVerdict::Ok {
            ranges,
            total_bytes,
        } = v
        {
            assert_eq!(ranges.len(), 1);
            assert_eq!(total_bytes, 1000);
        }
    }
}
