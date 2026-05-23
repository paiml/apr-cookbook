//! # apr import --shards — Sharding Plan Validator
//!
//! `apr import <DIR>` resolves multi-file shards (e.g., `model-001-of-N.bin`).
//! Constraints: filenames must follow the pattern; total count matches
//! N; no duplicates; no gaps. This recipe builds the validator.
//!
//! Demonstrates the **IMP.6** recipe for PMAT-115 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender IMP-001 + HuggingFace shard naming convention
//!
//! Run with: cargo run --example cli_import_sharding_plan_validator
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum ShardVerdict {
    Ok,
    InvalidPattern { name: String },
    InconsistentTotal { found: u32, declared: u32 },
    Duplicate { index: u32 },
    Gap { missing: u32 },
    Empty,
}

pub fn validate(filenames: &[&str]) -> ShardVerdict {
    if filenames.is_empty() {
        return ShardVerdict::Empty;
    }
    let mut declared_total: Option<u32> = None;
    let mut seen: HashSet<u32> = HashSet::new();
    for name in filenames {
        let Some((idx, total)) = parse_shard_name(name) else {
            return ShardVerdict::InvalidPattern {
                name: (*name).to_string(),
            };
        };
        if let Some(t) = declared_total {
            if t != total {
                return ShardVerdict::InconsistentTotal {
                    found: total,
                    declared: t,
                };
            }
        } else {
            declared_total = Some(total);
        }
        if !seen.insert(idx) {
            return ShardVerdict::Duplicate { index: idx };
        }
    }
    if let Some(total) = declared_total {
        for i in 1..=total {
            if !seen.contains(&i) {
                return ShardVerdict::Gap { missing: i };
            }
        }
    }
    ShardVerdict::Ok
}

fn parse_shard_name(name: &str) -> Option<(u32, u32)> {
    // Accept `model-001-of-005.bin` or `weights-001-of-005.safetensors`.
    let stem = name.rsplit_once('.').map_or(name, |(s, _)| s);
    let (left, total_str) = stem.rsplit_once("-of-")?;
    let (_, idx_str) = left.rsplit_once('-')?;
    Some((idx_str.parse().ok()?, total_str.parse().ok()?))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_sharding_plan_validator")?;

    let valid = [
        "model-001-of-003.bin",
        "model-002-of-003.bin",
        "model-003-of-003.bin",
    ];
    println!("valid: {:?}", validate(&valid));
    let gap = ["model-001-of-003.bin", "model-003-of-003.bin"];
    println!("gap:   {:?}", validate(&gap));
    let dup = ["model-001-of-003.bin", "model-001-of-003.bin"];
    println!("dup:   {:?}", validate(&dup));
    let mismatch = ["model-001-of-003.bin", "model-002-of-005.bin"];
    println!("mix:   {:?}", validate(&mismatch));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_in_order_passes() {
        let names = [
            "model-001-of-003.bin",
            "model-002-of-003.bin",
            "model-003-of-003.bin",
        ];
        assert_eq!(validate(&names), ShardVerdict::Ok);
    }

    #[test]
    fn out_of_order_passes() {
        // Order shouldn't matter for validation.
        let names = [
            "model-003-of-003.bin",
            "model-001-of-003.bin",
            "model-002-of-003.bin",
        ];
        assert_eq!(validate(&names), ShardVerdict::Ok);
    }

    #[test]
    fn gap_detected() {
        let names = ["model-001-of-003.bin", "model-003-of-003.bin"];
        let v = validate(&names);
        assert!(matches!(v, ShardVerdict::Gap { missing: 2 }));
    }

    #[test]
    fn duplicate_detected() {
        let names = ["model-001-of-003.bin", "model-001-of-003.bin"];
        let v = validate(&names);
        assert!(matches!(v, ShardVerdict::Duplicate { index: 1 }));
    }

    #[test]
    fn inconsistent_total_detected() {
        let names = ["model-001-of-003.bin", "model-002-of-005.bin"];
        let v = validate(&names);
        assert!(matches!(v, ShardVerdict::InconsistentTotal { .. }));
    }

    #[test]
    fn invalid_pattern_rejected() {
        let names = ["model.bin"]; // no shard suffix
        let v = validate(&names);
        assert!(matches!(v, ShardVerdict::InvalidPattern { .. }));
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(validate(&[]), ShardVerdict::Empty);
    }

    #[test]
    fn safetensors_extension_accepted() {
        let names = [
            "weights-001-of-002.safetensors",
            "weights-002-of-002.safetensors",
        ];
        assert_eq!(validate(&names), ShardVerdict::Ok);
    }

    #[test]
    fn single_shard_passes() {
        let names = ["model-001-of-001.bin"];
        assert_eq!(validate(&names), ShardVerdict::Ok);
    }
}
