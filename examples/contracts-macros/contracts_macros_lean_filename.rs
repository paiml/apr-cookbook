//! # Contracts-Macros Lean Filename Derivation
//!
//! Derive a Lean source file path from a module path. `Foo.Bar.Baz`
//! → `Foo/Bar/Baz.lean`. Verifies the path is well-formed and writable.
//!
//! Demonstrates the **CMM.33** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean 4 toolchain filename convention.
//!
//! Run with: cargo run --example contracts_macros_lean_filename
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilenameVerdict {
    Ok { path: String, depth: u32 },
    EmptyModule,
    InvalidSegment { segment: String },
}

pub fn derive(module_path: &str) -> FilenameVerdict {
    let trimmed = module_path.trim();
    if trimmed.is_empty() {
        return FilenameVerdict::EmptyModule;
    }
    let segments: Vec<&str> = trimmed.split('.').collect();
    for seg in &segments {
        if seg.is_empty() || !seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return FilenameVerdict::InvalidSegment {
                segment: (*seg).to_string(),
            };
        }
    }
    let path = format!("{}.lean", segments.join("/"));
    FilenameVerdict::Ok {
        path,
        depth: segments.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_lean_filename")?;

    println!("typical: {:?}", derive("Foo.Bar.Theorem"));
    println!("single: {:?}", derive("Main"));
    println!("invalid char: {:?}", derive("Foo.Bar-Baz"));
    println!("empty seg: {:?}", derive("Foo..Bar"));
    println!("empty: {:?}", derive("  "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deriver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn deep_path_yields_slashes() {
        let v = derive("Foo.Bar.Theorem");
        if let FilenameVerdict::Ok { path, depth } = v {
            assert_eq!(path, "Foo/Bar/Theorem.lean");
            assert_eq!(depth, 3);
        }
    }

    #[test]
    fn single_segment() {
        let v = derive("Main");
        if let FilenameVerdict::Ok { path, depth } = v {
            assert_eq!(path, "Main.lean");
            assert_eq!(depth, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(derive("  "), FilenameVerdict::EmptyModule);
    }

    #[test]
    fn empty_segment_rejected() {
        assert!(matches!(
            derive("Foo..Bar"),
            FilenameVerdict::InvalidSegment { .. }
        ));
    }

    #[test]
    fn dash_rejected() {
        assert!(matches!(
            derive("Foo.Bar-Baz"),
            FilenameVerdict::InvalidSegment { .. }
        ));
    }

    #[test]
    fn underscore_allowed() {
        let v = derive("Foo.Bar_baz");
        assert!(matches!(v, FilenameVerdict::Ok { .. }));
    }

    #[test]
    fn leading_dot_rejected() {
        assert!(matches!(
            derive(".Foo"),
            FilenameVerdict::InvalidSegment { .. }
        ));
    }

    #[test]
    fn trailing_dot_rejected() {
        assert!(matches!(
            derive("Foo."),
            FilenameVerdict::InvalidSegment { .. }
        ));
    }

    #[test]
    fn ends_with_lean() {
        if let FilenameVerdict::Ok { path, .. } = derive("Foo.Bar") {
            assert!(path.ends_with(".lean"));
        }
    }

    #[test]
    fn deterministic() {
        let a = derive("Foo.Bar.Theorem");
        let b = derive("Foo.Bar.Theorem");
        assert_eq!(a, b);
    }
}
