//! # WASM Component-Model Export Validator
//!
//! Component Model exports are typed (interface types). Each export
//! must list:
//!   - name (kebab-case, ≤ 64 chars)
//!   - kind (function | resource | type)
//!   - signature (params + result type)
//!
//! Validator returns Ok / NamingViolation / DuplicateName / EmptyExports.
//!
//! Demonstrates the **WASM.24** recipe for PMAT-151 (wasm round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Component Model spec (Bytecode Alliance).
//!
//! Run with: cargo run --example wasm_component_export
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportKind {
    Function,
    Resource,
    Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub kind: ExportKind,
}

#[derive(Debug, PartialEq)]
pub enum ExportVerdict {
    Ok { count: usize },
    EmptyExports,
    NamingViolation { name: String, reason: &'static str },
    DuplicateName { name: String },
}

pub fn validate(exports: &[Export]) -> ExportVerdict {
    if exports.is_empty() {
        return ExportVerdict::EmptyExports;
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for e in exports {
        if e.name.is_empty() {
            return ExportVerdict::NamingViolation {
                name: e.name.clone(),
                reason: "empty name",
            };
        }
        if e.name.len() > MAX_NAME_LEN {
            return ExportVerdict::NamingViolation {
                name: e.name.clone(),
                reason: "exceeds 64 char limit",
            };
        }
        if !is_kebab_case(&e.name) {
            return ExportVerdict::NamingViolation {
                name: e.name.clone(),
                reason: "must be kebab-case",
            };
        }
        if !seen.insert(&e.name) {
            return ExportVerdict::DuplicateName {
                name: e.name.clone(),
            };
        }
    }
    ExportVerdict::Ok {
        count: exports.len(),
    }
}

fn is_kebab_case(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !s.starts_with('-')
        && !s.ends_with('-')
        && !s.contains("--")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_component_export")?;

    let exports = vec![
        Export {
            name: "predict".to_string(),
            kind: ExportKind::Function,
        },
        Export {
            name: "model-handle".to_string(),
            kind: ExportKind::Resource,
        },
    ];
    println!("valid: {:?}", validate(&exports));

    let bad = vec![Export {
        name: "PredictFn".to_string(),
        kind: ExportKind::Function,
    }];
    println!("PascalCase: {:?}", validate(&bad));

    let dup = vec![
        Export {
            name: "x".to_string(),
            kind: ExportKind::Function,
        },
        Export {
            name: "x".to_string(),
            kind: ExportKind::Type,
        },
    ];
    println!("duplicate: {:?}", validate(&dup));
    println!("empty: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export(name: &str, kind: ExportKind) -> Export {
        Export {
            name: name.to_string(),
            kind,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_kebab_case_ok() {
        let v = validate(&[export("predict", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::Ok { .. }));
    }

    #[test]
    fn pascal_case_rejected() {
        let v = validate(&[export("PredictFn", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn underscore_rejected() {
        let v = validate(&[export("predict_fn", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn duplicate_rejected() {
        let v = validate(&[
            export("x", ExportKind::Function),
            export("x", ExportKind::Type),
        ]);
        assert!(matches!(v, ExportVerdict::DuplicateName { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[]), ExportVerdict::EmptyExports);
    }

    #[test]
    fn empty_name_rejected() {
        let v = validate(&[export("", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn too_long_name_rejected() {
        let long = "a".repeat(65);
        let v = validate(&[export(&long, ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn leading_dash_rejected() {
        let v = validate(&[export("-foo", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn trailing_dash_rejected() {
        let v = validate(&[export("foo-", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn double_dash_rejected() {
        let v = validate(&[export("foo--bar", ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::NamingViolation { .. }));
    }

    #[test]
    fn count_returned() {
        let v = validate(&[
            export("a", ExportKind::Function),
            export("b", ExportKind::Resource),
            export("c", ExportKind::Type),
        ]);
        if let ExportVerdict::Ok { count } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn at_max_length_ok() {
        let max = "a".repeat(MAX_NAME_LEN);
        let v = validate(&[export(&max, ExportKind::Function)]);
        assert!(matches!(v, ExportVerdict::Ok { .. }));
    }
}
