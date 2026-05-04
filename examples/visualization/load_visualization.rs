//! # Visualization Config Validator
//!
//! Loads every `.yaml` and `.prs` file under `examples/visualization/`, parses
//! each as YAML, and asserts it is well-formed. Acts as the single IIUR-graded
//! Rust artifact for the entire visualization corpus (Class 2 Strategy B per
//! `docs/specifications/centralize-cookbooks/iiur-conformance.md`).
//!
//! Contract: contracts/recipe-iiur-config-v1.yaml
//! Citation: Tufte, E. R. (2001). The Visual Display of Quantitative Information (2nd ed). Graphics Press. ISBN: 978-1930824133
//!
//! Run with: cargo run --example load_visualization
//!
//! Migrated from presentar by PMAT-067 (centralize-cookbooks).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::{CookbookError, Result};
use std::fs;
use std::path::PathBuf;

const SUBDIRS: &[&str] = &["ald", "apr", "charts", "dashboards", "edge_cases", "prs"];

fn config_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("visualization");
    for sub in SUBDIRS {
        let dir = root.join(sub);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("yaml" | "prs")
            ) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn validate_one(path: &PathBuf) -> Result<()> {
    let content = fs::read_to_string(path)?;
    // Both .yaml and .prs are YAML-encoded.
    let _: serde_yaml::Value =
        serde_yaml::from_str(&content).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("load_visualization")?;
    let files = config_files();
    println!("validating {} visualization configs", files.len());
    for f in &files {
        validate_one(f)?;
        println!(
            "  OK: {}",
            f.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .unwrap_or(f)
                .display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_28_configs() {
        let n = config_files().len();
        assert!(n >= 28, "expected >= 28 configs, found {n}");
    }

    #[test]
    fn all_configs_parse() {
        for f in &config_files() {
            validate_one(f).unwrap_or_else(|e| panic!("config {} failed: {e}", f.display()));
        }
    }

    #[test]
    fn wrapper_runs() {
        main().expect("validator should run successfully");
    }

    #[test]
    fn each_subdir_has_at_least_one_config() {
        let files = config_files();
        for sub in SUBDIRS {
            let n = files
                .iter()
                .filter(|p| {
                    p.parent()
                        .and_then(|d| d.file_name())
                        .is_some_and(|d| d == *sub)
                })
                .count();
            assert!(n >= 1, "subdir {sub} has no configs");
        }
    }
}
