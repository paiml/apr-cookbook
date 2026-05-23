//! # Contracts-Macros Lean Module Binding
//!
//! For each YAML obligation declaring `lean.module: "Foo.Bar.Theorem"`,
//! verify the module path is well-formed: dot-separated, non-empty
//! segments, capitalized first char of each segment.
//!
//! Demonstrates the **CMM.14** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean 4 namespacing convention.
//!
//! Run with: cargo run --example contracts_macros_module_binding
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ModuleVerdict {
    Ok { segments: Vec<String> },
    EmptyPath,
    EmptySegment,
    InvalidStartChar { segment: String },
    InvalidChar { segment: String },
}

pub fn validate(module_path: &str) -> ModuleVerdict {
    let trimmed = module_path.trim();
    if trimmed.is_empty() {
        return ModuleVerdict::EmptyPath;
    }
    let segments: Vec<&str> = trimmed.split('.').collect();
    for seg in &segments {
        if seg.is_empty() {
            return ModuleVerdict::EmptySegment;
        }
        let first = seg.chars().next().unwrap_or('_');
        if !first.is_ascii_uppercase() {
            return ModuleVerdict::InvalidStartChar {
                segment: (*seg).to_string(),
            };
        }
        if !seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return ModuleVerdict::InvalidChar {
                segment: (*seg).to_string(),
            };
        }
    }
    ModuleVerdict::Ok {
        segments: segments.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_module_binding")?;

    println!("ok: {:?}", validate("Foo.Bar.Theorem"));
    println!("single: {:?}", validate("Main"));
    println!("empty: {:?}", validate("  "));
    println!("trailing dot: {:?}", validate("Foo."));
    println!("lowercase start: {:?}", validate("foo.Bar"));
    println!("invalid char: {:?}", validate("Foo.Bar-Theorem"));
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
    fn well_formed_path_ok() {
        let v = validate("Foo.Bar.Theorem");
        if let ModuleVerdict::Ok { segments } = v {
            assert_eq!(segments, vec!["Foo", "Bar", "Theorem"]);
        }
    }

    #[test]
    fn single_segment_ok() {
        let v = validate("Main");
        if let ModuleVerdict::Ok { segments } = v {
            assert_eq!(segments, vec!["Main"]);
        }
    }

    #[test]
    fn empty_path_rejected() {
        assert_eq!(validate("  "), ModuleVerdict::EmptyPath);
    }

    #[test]
    fn trailing_dot_empty_segment() {
        assert_eq!(validate("Foo."), ModuleVerdict::EmptySegment);
    }

    #[test]
    fn leading_dot_empty_segment() {
        assert_eq!(validate(".Foo"), ModuleVerdict::EmptySegment);
    }

    #[test]
    fn lowercase_start_rejected() {
        let v = validate("foo.Bar");
        assert!(matches!(v, ModuleVerdict::InvalidStartChar { .. }));
    }

    #[test]
    fn invalid_char_rejected() {
        let v = validate("Foo.Bar-Theorem");
        assert!(matches!(v, ModuleVerdict::InvalidChar { .. }));
    }

    #[test]
    fn underscore_allowed() {
        let v = validate("Foo.Bar_baz");
        assert!(matches!(v, ModuleVerdict::Ok { .. }));
    }

    #[test]
    fn digits_allowed_after_first() {
        let v = validate("Foo2.Bar3");
        assert!(matches!(v, ModuleVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let a = validate("Foo.Bar");
        let b = validate("Foo.Bar");
        assert_eq!(a, b);
    }
}
