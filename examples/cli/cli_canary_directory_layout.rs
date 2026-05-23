//! # apr canary — Directory Layout Convention
//!
//! `apr canary create` writes snapshots under `.apr/canaries/<name>.json`
//! by default; `apr canary check` reads them from the same path. This
//! recipe documents the layout and asserts the contract: paths are
//! deterministic per (name, root), and conflicting entries are an error
//! (no silent overwrite without `--force`).
//!
//! Demonstrates the **CANARY.6** recipe for PMAT-100 (apr canary coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CANARY-003
//!
//! Run with: cargo run --example cli_canary_directory_layout
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::path::PathBuf;

pub fn snapshot_path(root: &str, name: &str) -> PathBuf {
    PathBuf::from(root)
        .join(".apr")
        .join("canaries")
        .join(format!("{name}.json"))
}

#[derive(Debug, PartialEq)]
pub enum WriteVerdict {
    Ok,
    NameConflict,
    InvalidName,
}

pub fn validate_write(name: &str, exists_on_disk: bool, force: bool) -> WriteVerdict {
    if name.is_empty()
        || name
            .chars()
            .any(|c| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
    {
        return WriteVerdict::InvalidName;
    }
    if exists_on_disk && !force {
        return WriteVerdict::NameConflict;
    }
    WriteVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_canary_directory_layout")?;

    println!("path resolution:");
    for (root, name) in [
        ("/repo", "math"),
        (".", "smoke-2plus2"),
        ("/tmp", "v1_alpha"),
    ] {
        let p = snapshot_path(root, name);
        println!("  ({root:>8}, {name:>15}) → {}", p.display());
    }

    println!("\nwrite verdicts:");
    for (label, name, exists, force) in [
        ("happy", "math", false, false),
        ("conflict", "math", true, false),
        ("conflict force", "math", true, true),
        ("invalid name", "math/sub", false, false),
    ] {
        println!(
            "  {label:>16}  →  {:?}",
            validate_write(name, exists, force)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn snapshot_path_is_deterministic() {
        let a = snapshot_path("/repo", "math");
        let b = snapshot_path("/repo", "math");
        assert_eq!(a, b);
    }

    #[test]
    fn snapshot_path_uses_dot_apr_canaries_prefix() {
        let p = snapshot_path("/repo", "math");
        let s = p.to_string_lossy();
        assert!(s.contains(".apr/canaries/math.json"));
    }

    #[test]
    fn happy_write_passes() {
        assert_eq!(validate_write("math", false, false), WriteVerdict::Ok);
    }

    #[test]
    fn name_conflict_without_force_rejected() {
        // Critical: don't silently overwrite existing canaries.
        assert_eq!(
            validate_write("math", true, false),
            WriteVerdict::NameConflict
        );
    }

    #[test]
    fn force_allows_overwrite() {
        assert_eq!(validate_write("math", true, true), WriteVerdict::Ok);
    }

    #[test]
    fn invalid_name_rejected_independent_of_existence() {
        assert_eq!(
            validate_write("math/sub", false, false),
            WriteVerdict::InvalidName
        );
        assert_eq!(
            validate_write("math/sub", true, true),
            WriteVerdict::InvalidName
        );
    }

    #[test]
    fn empty_name_rejected() {
        assert_eq!(validate_write("", false, false), WriteVerdict::InvalidName);
    }
}
