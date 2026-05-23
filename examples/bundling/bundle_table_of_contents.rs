//! # Bundle Table-of-Contents Indexer
//!
//! TOC entry: (name, offset, length). Build sorted-by-offset table from
//! a list of (name, length) sequential allocations. This recipe builds
//! the indexer + the binary-search lookup-by-offset.
//!
//! Demonstrates the **BUNDLE.15** recipe for PMAT-133 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ZIP file central directory layout.
//!
//! Run with: cargo run --example bundle_table_of_contents
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TocEntry {
    pub name: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, PartialEq)]
pub enum BuildVerdict {
    Ok(Vec<TocEntry>),
    EmptyAllocations,
    DuplicateName { name: String },
    OverflowOnOffset,
}

pub fn build(allocations: &[(&str, u64)]) -> BuildVerdict {
    if allocations.is_empty() {
        return BuildVerdict::EmptyAllocations;
    }
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut entries = Vec::with_capacity(allocations.len());
    let mut offset = 0u64;
    for (name, length) in allocations {
        if !seen.insert(name) {
            return BuildVerdict::DuplicateName {
                name: (*name).to_string(),
            };
        }
        let new_end = offset.checked_add(*length);
        if new_end.is_none() {
            return BuildVerdict::OverflowOnOffset;
        }
        entries.push(TocEntry {
            name: (*name).to_string(),
            offset,
            length: *length,
        });
        offset = new_end.unwrap();
    }
    BuildVerdict::Ok(entries)
}

pub fn lookup_by_offset(entries: &[TocEntry], offset: u64) -> Option<&TocEntry> {
    let idx = entries
        .binary_search_by(|e| {
            if offset < e.offset {
                std::cmp::Ordering::Greater
            } else if offset >= e.offset + e.length {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()?;
    Some(&entries[idx])
}

pub fn total_size(entries: &[TocEntry]) -> u64 {
    entries.iter().map(|e| e.length).sum()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_table_of_contents")?;

    let alloc = [
        ("embed.weight", 1000u64),
        ("layer.0", 500),
        ("layer.1", 500),
    ];
    let v = build(&alloc);
    println!("toc: {v:?}");
    if let BuildVerdict::Ok(entries) = &v {
        println!("offset 1200 → {:?}", lookup_by_offset(entries, 1200));
        println!("total size: {}", total_size(entries));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_toc_built_correctly() {
        let alloc = [("a", 100u64), ("b", 200), ("c", 300)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].offset, 0);
            assert_eq!(entries[1].offset, 100);
            assert_eq!(entries[2].offset, 300);
        }
    }

    #[test]
    fn empty_allocations_rejected() {
        assert_eq!(build(&[]), BuildVerdict::EmptyAllocations);
    }

    #[test]
    fn duplicate_name_rejected() {
        let alloc = [("a", 100u64), ("a", 200)];
        let v = build(&alloc);
        assert!(matches!(v, BuildVerdict::DuplicateName { .. }));
    }

    #[test]
    fn overflow_rejected() {
        let alloc = [("a", u64::MAX), ("b", 1u64)];
        assert_eq!(build(&alloc), BuildVerdict::OverflowOnOffset);
    }

    #[test]
    fn lookup_within_entry_returns_it() {
        let alloc = [("a", 100u64), ("b", 200), ("c", 300)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            // Offset 250 falls in "b" (offset 100, len 200, range 100..300).
            let e = lookup_by_offset(&entries, 250);
            assert_eq!(e.unwrap().name, "b");
        }
    }

    #[test]
    fn lookup_at_entry_start_returns_it() {
        let alloc = [("a", 100u64), ("b", 200)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            let e = lookup_by_offset(&entries, 100);
            assert_eq!(e.unwrap().name, "b");
        }
    }

    #[test]
    fn lookup_past_end_yields_none() {
        let alloc = [("a", 100u64)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            assert!(lookup_by_offset(&entries, 1000).is_none());
        }
    }

    #[test]
    fn lookup_at_zero_returns_first() {
        let alloc = [("a", 100u64), ("b", 200)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            let e = lookup_by_offset(&entries, 0);
            assert_eq!(e.unwrap().name, "a");
        }
    }

    #[test]
    fn total_size_sums_lengths() {
        let alloc = [("a", 100u64), ("b", 200), ("c", 300)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            assert_eq!(total_size(&entries), 600);
        }
    }

    #[test]
    fn zero_length_entry_handled() {
        let alloc = [("a", 0u64), ("b", 100)];
        if let BuildVerdict::Ok(entries) = build(&alloc) {
            assert_eq!(entries[0].length, 0);
            assert_eq!(entries[1].offset, 0);
        }
    }
}
