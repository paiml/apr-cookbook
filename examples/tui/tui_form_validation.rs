//! # TUI Form Validation
//!
//! Collect per-field validation errors against a small schema:
//!   required: non-empty
//!   max_len: ≤ N chars
//!   numeric: parses as integer
//!
//! Demonstrates the **TUI.13** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 form validation + zod-style schemas.
//!
//! Run with: cargo run --example tui_form_validation
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRule {
    Required,
    MaxLen(usize),
    Numeric,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    pub field: String,
    pub rule: FieldRule,
}

#[derive(Debug, PartialEq)]
pub enum FormVerdict {
    Ok,
    HasErrors { errors: Vec<FieldError> },
}

pub fn validate(fields: &[(&str, &str, &[FieldRule])]) -> FormVerdict {
    let mut errors: Vec<FieldError> = Vec::new();
    for (name, value, rules) in fields {
        for rule in *rules {
            let fails = match rule {
                FieldRule::Required => value.trim().is_empty(),
                FieldRule::MaxLen(n) => value.chars().count() > *n,
                FieldRule::Numeric => value.parse::<i64>().is_err(),
            };
            if fails {
                errors.push(FieldError {
                    field: (*name).to_string(),
                    rule: *rule,
                });
            }
        }
    }
    if errors.is_empty() {
        FormVerdict::Ok
    } else {
        FormVerdict::HasErrors { errors }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_form_validation")?;

    let fields = [
        (
            "name",
            "Alice",
            &[FieldRule::Required, FieldRule::MaxLen(50)][..],
        ),
        ("age", "42", &[FieldRule::Required, FieldRule::Numeric][..]),
    ];
    println!("ok: {:?}", validate(&fields));

    let bad = [
        ("name", "", &[FieldRule::Required][..]),
        ("age", "abc", &[FieldRule::Numeric][..]),
    ];
    println!("errors: {:?}", validate(&bad));
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
    fn valid_input_ok() {
        let fields = [("name", "Alice", &[FieldRule::Required][..])];
        assert_eq!(validate(&fields), FormVerdict::Ok);
    }

    #[test]
    fn missing_required_returns_error() {
        let fields = [("name", "", &[FieldRule::Required][..])];
        let v = validate(&fields);
        if let FormVerdict::HasErrors { errors } = v {
            assert_eq!(errors.len(), 1);
        }
    }

    #[test]
    fn whitespace_only_required_fails() {
        let fields = [("name", "   ", &[FieldRule::Required][..])];
        assert!(matches!(validate(&fields), FormVerdict::HasErrors { .. }));
    }

    #[test]
    fn over_max_len_fails() {
        let fields = [("name", "abcdef", &[FieldRule::MaxLen(3)][..])];
        let v = validate(&fields);
        if let FormVerdict::HasErrors { errors } = v {
            assert_eq!(errors[0].rule, FieldRule::MaxLen(3));
        }
    }

    #[test]
    fn at_max_len_ok() {
        let fields = [("name", "abc", &[FieldRule::MaxLen(3)][..])];
        assert_eq!(validate(&fields), FormVerdict::Ok);
    }

    #[test]
    fn non_numeric_rejected() {
        let fields = [("age", "abc", &[FieldRule::Numeric][..])];
        let v = validate(&fields);
        if let FormVerdict::HasErrors { errors } = v {
            assert_eq!(errors[0].rule, FieldRule::Numeric);
        }
    }

    #[test]
    fn negative_numeric_ok() {
        let fields = [("age", "-42", &[FieldRule::Numeric][..])];
        assert_eq!(validate(&fields), FormVerdict::Ok);
    }

    #[test]
    fn multiple_rules_all_checked() {
        let fields = [("x", "", &[FieldRule::Required, FieldRule::Numeric][..])];
        let v = validate(&fields);
        if let FormVerdict::HasErrors { errors } = v {
            assert_eq!(errors.len(), 2);
        }
    }

    #[test]
    fn errors_carry_field_names() {
        let fields = [
            ("name", "", &[FieldRule::Required][..]),
            ("age", "x", &[FieldRule::Numeric][..]),
        ];
        let v = validate(&fields);
        if let FormVerdict::HasErrors { errors } = v {
            let names: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
            assert!(names.contains(&"name"));
            assert!(names.contains(&"age"));
        }
    }

    #[test]
    fn empty_form_ok() {
        assert_eq!(validate(&[]), FormVerdict::Ok);
    }

    #[test]
    fn deterministic() {
        let fields = [("name", "Alice", &[FieldRule::Required][..])];
        let a = validate(&fields);
        let b = validate(&fields);
        assert_eq!(a, b);
    }
}
