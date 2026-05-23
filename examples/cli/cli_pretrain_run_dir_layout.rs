//! # apr pretrain — `--run-dir` Output Layout
//!
//! `apr pretrain --run-dir <DIR>` deposits artifacts under `<DIR>/ckpt/`
//! (checkpoints + metadata). This recipe documents the layout and asserts
//! the contract: directory must be writable + empty (or missing); paths
//! follow `<run_dir>/ckpt/step-<N>.apr` naming for diff-able CI logs.
//!
//! Demonstrates the **PRETRAIN.6** recipe for PMAT-104 (apr pretrain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-TWO-001
//!
//! Run with: cargo run --example cli_pretrain_run_dir_layout
//!
//! Added by PMAT-104 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::path::PathBuf;

pub fn checkpoint_path(run_dir: &str, step: u32) -> PathBuf {
    PathBuf::from(run_dir)
        .join("ckpt")
        .join(format!("step-{step:08}.apr"))
}

pub fn metadata_path(run_dir: &str) -> PathBuf {
    PathBuf::from(run_dir).join("ckpt").join("metadata.json")
}

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok,
    EmptyRunDir,
    NonEmptyRunDir,
}

pub fn validate_run_dir(path: &str, exists_and_empty: Option<bool>) -> LayoutVerdict {
    if path.is_empty() {
        return LayoutVerdict::EmptyRunDir;
    }
    match exists_and_empty {
        None => LayoutVerdict::Ok, // doesn't exist yet; will create
        Some(true) => LayoutVerdict::Ok,
        Some(false) => LayoutVerdict::NonEmptyRunDir,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pretrain_run_dir_layout")?;

    println!("step paths:");
    for s in [0u32, 100, 1_000, 100_000] {
        println!(
            "  step {s:>6} → {}",
            checkpoint_path("/runs/r1", s).display()
        );
    }
    println!("\nmetadata: {}", metadata_path("/runs/r1").display());

    println!("\nvalidation:");
    for (label, path, exists) in [
        ("missing dir", "/new/run", None),
        ("empty dir", "/runs/empty", Some(true)),
        ("dirty dir", "/runs/dirty", Some(false)),
        ("empty path", "", None),
    ] {
        println!("  {label:>15}  →  {:?}", validate_run_dir(path, exists));
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
    fn step_path_uses_8_digit_zero_padding() {
        // 8-digit suffix sorts naturally up to 100M steps.
        let p = checkpoint_path("/r", 1234);
        assert!(p.to_string_lossy().ends_with("step-00001234.apr"));
    }

    #[test]
    fn step_path_under_ckpt_subdir() {
        let p = checkpoint_path("/r", 0);
        assert!(p.to_string_lossy().contains("/ckpt/"));
    }

    #[test]
    fn metadata_path_canonical() {
        let p = metadata_path("/r");
        assert!(p.to_string_lossy().ends_with("/ckpt/metadata.json"));
    }

    #[test]
    fn empty_run_dir_string_rejected() {
        assert_eq!(validate_run_dir("", None), LayoutVerdict::EmptyRunDir);
    }

    #[test]
    fn missing_dir_passes() {
        // Nonexistent dir is fine — the run will create it.
        assert_eq!(validate_run_dir("/new/run", None), LayoutVerdict::Ok);
    }

    #[test]
    fn empty_existing_dir_passes() {
        assert_eq!(
            validate_run_dir("/runs/empty", Some(true)),
            LayoutVerdict::Ok
        );
    }

    #[test]
    fn nonempty_existing_dir_rejected() {
        // Must NOT clobber an existing run dir.
        assert_eq!(
            validate_run_dir("/runs/dirty", Some(false)),
            LayoutVerdict::NonEmptyRunDir
        );
    }

    #[test]
    fn step_zero_padded_correctly() {
        // Edge: step 0 still gets padded.
        let p = checkpoint_path("/r", 0);
        assert!(p.to_string_lossy().ends_with("step-00000000.apr"));
    }
}
