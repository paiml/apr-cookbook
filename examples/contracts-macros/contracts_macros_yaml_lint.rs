//! # Contracts-Macros YAML Lint
//!
//! Static lint for IIUR contract YAMLs: each equation must declare
//! preconditions, postconditions, lean_theorem, tolerance. Returns
//! the first missing field per equation.
//!
//! Demonstrates the **CMM.18** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: provable_contracts mandatory-fields convention.
//!
//! Run with: cargo run --example contracts_macros_yaml_lint
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationFields {
    pub name: String,
    pub has_preconds: bool,
    pub has_postconds: bool,
    pub has_lean_theorem: bool,
    pub has_tolerance: bool,
}

#[derive(Debug, PartialEq)]
pub enum LintVerdict {
    Ok {
        equation_count: u32,
    },
    MissingField {
        equation: String,
        field: &'static str,
    },
    EmptyContract,
}

pub fn lint(equations: &[EquationFields]) -> LintVerdict {
    if equations.is_empty() {
        return LintVerdict::EmptyContract;
    }
    for eq in equations {
        if !eq.has_preconds {
            return LintVerdict::MissingField {
                equation: eq.name.clone(),
                field: "preconditions",
            };
        }
        if !eq.has_postconds {
            return LintVerdict::MissingField {
                equation: eq.name.clone(),
                field: "postconditions",
            };
        }
        if !eq.has_lean_theorem {
            return LintVerdict::MissingField {
                equation: eq.name.clone(),
                field: "lean_theorem",
            };
        }
        if !eq.has_tolerance {
            return LintVerdict::MissingField {
                equation: eq.name.clone(),
                field: "tolerance",
            };
        }
    }
    LintVerdict::Ok {
        equation_count: equations.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_lint")?;

    let ok = vec![EquationFields {
        name: "norm".to_string(),
        has_preconds: true,
        has_postconds: true,
        has_lean_theorem: true,
        has_tolerance: true,
    }];
    println!("ok: {:?}", lint(&ok));

    let missing = vec![EquationFields {
        name: "decode".to_string(),
        has_preconds: true,
        has_postconds: false,
        has_lean_theorem: true,
        has_tolerance: true,
    }];
    println!("missing post: {:?}", lint(&missing));
    println!("empty: {:?}", lint(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(name: &str) -> EquationFields {
        EquationFields {
            name: name.to_string(),
            has_preconds: true,
            has_postconds: true,
            has_lean_theorem: true,
            has_tolerance: true,
        }
    }

    #[test]
    fn linter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_passes() {
        let v = lint(&[complete("a")]);
        if let LintVerdict::Ok { equation_count } = v {
            assert_eq!(equation_count, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(lint(&[]), LintVerdict::EmptyContract);
    }

    #[test]
    fn missing_preconds_returned_first() {
        let mut e = complete("a");
        e.has_preconds = false;
        let v = lint(&[e]);
        if let LintVerdict::MissingField { field, .. } = v {
            assert_eq!(field, "preconditions");
        }
    }

    #[test]
    fn missing_postconds_returned() {
        let mut e = complete("a");
        e.has_postconds = false;
        let v = lint(&[e]);
        if let LintVerdict::MissingField { field, .. } = v {
            assert_eq!(field, "postconditions");
        }
    }

    #[test]
    fn missing_lean_returned() {
        let mut e = complete("a");
        e.has_lean_theorem = false;
        let v = lint(&[e]);
        if let LintVerdict::MissingField { field, .. } = v {
            assert_eq!(field, "lean_theorem");
        }
    }

    #[test]
    fn missing_tolerance_returned() {
        let mut e = complete("a");
        e.has_tolerance = false;
        let v = lint(&[e]);
        if let LintVerdict::MissingField { field, .. } = v {
            assert_eq!(field, "tolerance");
        }
    }

    #[test]
    fn first_failing_equation_returned() {
        let mut bad = complete("b");
        bad.has_preconds = false;
        let v = lint(&[complete("a"), bad, complete("c")]);
        if let LintVerdict::MissingField { equation, .. } = v {
            assert_eq!(equation, "b");
        }
    }

    #[test]
    fn multiple_complete_pass() {
        let v = lint(&[complete("a"), complete("b"), complete("c")]);
        if let LintVerdict::Ok { equation_count } = v {
            assert_eq!(equation_count, 3);
        }
    }

    #[test]
    fn precond_missing_short_circuits_others() {
        // Even if multiple fields are missing, returns precond first.
        let e = EquationFields {
            name: "a".to_string(),
            has_preconds: false,
            has_postconds: false,
            has_lean_theorem: false,
            has_tolerance: false,
        };
        let v = lint(&[e]);
        if let LintVerdict::MissingField { field, .. } = v {
            assert_eq!(field, "preconditions");
        }
    }

    #[test]
    fn deterministic() {
        let eq = vec![complete("a")];
        let a = lint(&eq);
        let b = lint(&eq);
        assert_eq!(a, b);
    }
}
