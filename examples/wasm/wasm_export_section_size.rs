//! # WASM Export-Section Size Budget
//!
//! Each export adds 4-12 bytes (length-prefixed name + index). For
//! cold-start budget, recommend ≤ 64 KiB total. Above 256 KiB, V8's
//! parser slows. This recipe estimates total bytes + tier.
//!
//! Demonstrates the **WASM.18** recipe for PMAT-142 (wasm round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: V8 WASM module-validation perf benchmarks.
//!
//! Run with: cargo run --example wasm_export_section_size
//!
//! Added by PMAT-142 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ENTRY_FIXED_BYTES: u64 = 4;
const SMALL_TIER_BYTES: u64 = 64 * 1024;
const MEDIUM_TIER_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionTier {
    Slim,
    Average,
    Bloated,
}

#[derive(Debug, PartialEq)]
pub enum ExportSizeVerdict {
    Ok { total_bytes: u64, tier: SectionTier },
    EmptyExports,
    NameTooLong { name_len: usize, max: usize },
}

const MAX_NAME_LEN: usize = 1024;

pub fn estimate(export_names: &[&str]) -> ExportSizeVerdict {
    if export_names.is_empty() {
        return ExportSizeVerdict::EmptyExports;
    }
    let mut total_bytes = 0u64;
    for n in export_names {
        if n.len() > MAX_NAME_LEN {
            return ExportSizeVerdict::NameTooLong {
                name_len: n.len(),
                max: MAX_NAME_LEN,
            };
        }
        total_bytes += ENTRY_FIXED_BYTES + n.len() as u64;
    }
    let tier = if total_bytes <= SMALL_TIER_BYTES {
        SectionTier::Slim
    } else if total_bytes <= MEDIUM_TIER_BYTES {
        SectionTier::Average
    } else {
        SectionTier::Bloated
    };
    ExportSizeVerdict::Ok { total_bytes, tier }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_export_section_size")?;

    let small = ["predict", "init", "memory"];
    println!("3 exports: {:?}", estimate(&small));

    // Synthesize 5000 short exports.
    let many: Vec<String> = (0..5000).map(|i| format!("export_{i}")).collect();
    let many_refs: Vec<&str> = many.iter().map(String::as_str).collect();
    println!("5000 exports: {:?}", estimate(&many_refs));

    // Synthesize a huge export count.
    let huge: Vec<String> = (0..30_000).map(|i| format!("export_{i:08}")).collect();
    let huge_refs: Vec<&str> = huge.iter().map(String::as_str).collect();
    println!("30k exports: {:?}", estimate(&huge_refs));

    println!("empty: {:?}", estimate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_export_set_slim() {
        let v = estimate(&["a", "b", "c"]);
        if let ExportSizeVerdict::Ok { tier, .. } = v {
            assert_eq!(tier, SectionTier::Slim);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(estimate(&[]), ExportSizeVerdict::EmptyExports);
    }

    #[test]
    fn name_too_long_rejected() {
        let long = "a".repeat(2000);
        let v = estimate(&[long.as_str()]);
        assert!(matches!(v, ExportSizeVerdict::NameTooLong { .. }));
    }

    #[test]
    fn moderate_exports_average() {
        // ~5000 exports × ~14 bytes = 70 KiB → Average.
        let names: Vec<String> = (0..5000).map(|i| format!("export_{i:04}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        if let ExportSizeVerdict::Ok { tier, .. } = estimate(&refs) {
            assert_eq!(tier, SectionTier::Average);
        }
    }

    #[test]
    fn many_exports_bloated() {
        // 30000 exports × ~17 bytes ≈ 510 KiB → Bloated.
        let names: Vec<String> = (0..30_000).map(|i| format!("export_{i:08}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        if let ExportSizeVerdict::Ok { tier, .. } = estimate(&refs) {
            assert_eq!(tier, SectionTier::Bloated);
        }
    }

    #[test]
    fn total_bytes_includes_overhead() {
        // 1 entry of name "ab" → 4 (overhead) + 2 (name) = 6 bytes.
        if let ExportSizeVerdict::Ok { total_bytes, .. } = estimate(&["ab"]) {
            assert_eq!(total_bytes, 6);
        }
    }

    #[test]
    fn longer_names_higher_total() {
        let short = estimate(&["a"]);
        let long = estimate(&["alongfunctionname"]);
        if let (
            ExportSizeVerdict::Ok { total_bytes: s, .. },
            ExportSizeVerdict::Ok { total_bytes: l, .. },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn boundary_at_64kib_slim() {
        // Exactly 64 KiB → still Slim.
        let count: u64 = 64 * 1024 / (ENTRY_FIXED_BYTES + 4); // each "test"
        let names: Vec<String> = (0..count).map(|_| "test".to_string()).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        if let ExportSizeVerdict::Ok { tier, .. } = estimate(&refs) {
            assert_eq!(tier, SectionTier::Slim);
        }
    }

    #[test]
    fn name_at_max_succeeds() {
        let max_name = "a".repeat(MAX_NAME_LEN);
        let v = estimate(&[max_name.as_str()]);
        assert!(matches!(v, ExportSizeVerdict::Ok { .. }));
    }

    #[test]
    fn just_above_max_rejected() {
        let too_long = "a".repeat(MAX_NAME_LEN + 1);
        assert!(matches!(
            estimate(&[too_long.as_str()]),
            ExportSizeVerdict::NameTooLong { .. }
        ));
    }
}
