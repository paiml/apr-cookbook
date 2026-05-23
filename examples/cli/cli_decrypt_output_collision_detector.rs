//! # apr decrypt --output — Path Collision Detector
//!
//! `apr decrypt <FILE> --output <PATH>` writes plaintext to PATH. To
//! avoid silently clobbering, the CLI requires either: (a) PATH does
//! not exist, OR (b) `--force` is set, OR (c) PATH ≠ input FILE
//! (in-place would lose data on partial decrypt). This recipe builds
//! the validator.
//!
//! Demonstrates the **DEC.4** recipe for PMAT-121 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DEC-001
//!
//! Run with: cargo run --example cli_decrypt_output_collision_detector
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CollisionVerdict {
    Ok,
    OutputExists { path: String },
    SamePathAsInput,
    EmptyPath,
}

pub fn check(
    input_path: &str,
    output_path: &str,
    output_exists: bool,
    force: bool,
) -> CollisionVerdict {
    if output_path.is_empty() {
        return CollisionVerdict::EmptyPath;
    }
    if input_path == output_path {
        return CollisionVerdict::SamePathAsInput;
    }
    if output_exists && !force {
        return CollisionVerdict::OutputExists {
            path: output_path.to_string(),
        };
    }
    CollisionVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_output_collision_detector")?;

    let cases = [
        ("model.apr", "out.bin", false, false),
        ("model.apr", "out.bin", true, false),
        ("model.apr", "out.bin", true, true),
        ("model.apr", "model.apr", false, true),
        ("model.apr", "", false, false),
    ];
    for (i, o, exists, f) in cases {
        println!(
            "in={i} out={o} exists={exists} force={f}  →  {:?}",
            check(i, o, exists, f)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nonexistent_output_passes() {
        assert_eq!(check("a.apr", "b.bin", false, false), CollisionVerdict::Ok);
    }

    #[test]
    fn existing_output_without_force_rejected() {
        let v = check("a.apr", "b.bin", true, false);
        assert!(matches!(v, CollisionVerdict::OutputExists { .. }));
    }

    #[test]
    fn existing_output_with_force_passes() {
        assert_eq!(check("a.apr", "b.bin", true, true), CollisionVerdict::Ok);
    }

    #[test]
    fn same_path_as_input_rejected_even_with_force() {
        // In-place decrypt would lose data on partial failure.
        assert_eq!(
            check("a.apr", "a.apr", false, true),
            CollisionVerdict::SamePathAsInput
        );
    }

    #[test]
    fn empty_output_rejected() {
        assert_eq!(
            check("a.apr", "", false, false),
            CollisionVerdict::EmptyPath
        );
    }

    #[test]
    fn empty_output_takes_priority_over_force() {
        assert_eq!(check("a.apr", "", true, true), CollisionVerdict::EmptyPath);
    }

    #[test]
    fn same_path_check_takes_priority_over_existence() {
        // Even if output exists+force, same-path is still rejected.
        assert_eq!(
            check("model.apr", "model.apr", true, true),
            CollisionVerdict::SamePathAsInput
        );
    }

    #[test]
    fn different_paths_with_force_pass_regardless_of_existence() {
        assert_eq!(check("a.apr", "b.bin", true, true), CollisionVerdict::Ok);
        assert_eq!(check("a.apr", "b.bin", false, true), CollisionVerdict::Ok);
    }
}
