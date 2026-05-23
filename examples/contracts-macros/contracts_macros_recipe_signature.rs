//! # Contracts-Macros Recipe Signature Diff
//!
//! Compute a stable signature of a recipe's input/output types and
//! diff against a previous snapshot. Reports type changes that are
//! breaking vs additive.
//!
//! Demonstrates the **CMM.23** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo-public-api semver-style API surface diff.
//!
//! Run with: cargo run --example contracts_macros_recipe_signature
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeSig {
    pub name: String,
    pub input_types: Vec<String>,
    pub output_type: String,
}

#[derive(Debug, PartialEq)]
pub enum SigDiffVerdict {
    Identical,
    Additive { added_input_count: u32 },
    Breaking { reason: &'static str },
    Renamed,
    EmptySignature,
}

pub fn diff(prev: &RecipeSig, current: &RecipeSig) -> SigDiffVerdict {
    if prev.name.is_empty() || current.name.is_empty() {
        return SigDiffVerdict::EmptySignature;
    }
    if prev.name != current.name {
        return SigDiffVerdict::Renamed;
    }
    if prev.output_type != current.output_type {
        return SigDiffVerdict::Breaking {
            reason: "output type changed",
        };
    }
    let prev_n = prev.input_types.len();
    let cur_n = current.input_types.len();
    // Existing inputs must match types exactly.
    let common = prev_n.min(cur_n);
    for i in 0..common {
        if prev.input_types[i] != current.input_types[i] {
            return SigDiffVerdict::Breaking {
                reason: "input type changed",
            };
        }
    }
    if cur_n < prev_n {
        return SigDiffVerdict::Breaking {
            reason: "input removed",
        };
    }
    if cur_n > prev_n {
        return SigDiffVerdict::Additive {
            added_input_count: (cur_n - prev_n) as u32,
        };
    }
    SigDiffVerdict::Identical
}

fn sig(name: &str, inputs: &[&str], output: &str) -> RecipeSig {
    RecipeSig {
        name: name.to_string(),
        input_types: inputs.iter().map(|s| (*s).to_string()).collect(),
        output_type: output.to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_signature")?;

    let v0 = sig("infer", &["Tensor", "u32"], "Result<Tensor>");
    let v1 = sig("infer", &["Tensor", "u32"], "Result<Tensor>");
    let v2 = sig("infer", &["Tensor", "u32", "Config"], "Result<Tensor>");
    let v3 = sig("infer", &["Tensor", "f64"], "Result<Tensor>");
    let v4 = sig("infer", &["Tensor", "u32"], "Tensor");

    println!("identical: {:?}", diff(&v0, &v1));
    println!("additive: {:?}", diff(&v0, &v2));
    println!("breaking input: {:?}", diff(&v0, &v3));
    println!("breaking output: {:?}", diff(&v0, &v4));
    println!(
        "renamed: {:?}",
        diff(&v0, &sig("predict", &["Tensor", "u32"], "Result<Tensor>"))
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> RecipeSig {
        sig("infer", &["Tensor", "u32"], "Result<Tensor>")
    }

    #[test]
    fn diffr_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_signatures() {
        assert_eq!(diff(&base(), &base()), SigDiffVerdict::Identical);
    }

    #[test]
    fn additive_input_recognized() {
        let v = diff(
            &base(),
            &sig("infer", &["Tensor", "u32", "Config"], "Result<Tensor>"),
        );
        if let SigDiffVerdict::Additive { added_input_count } = v {
            assert_eq!(added_input_count, 1);
        }
    }

    #[test]
    fn input_type_change_breaking() {
        let v = diff(&base(), &sig("infer", &["Tensor", "f64"], "Result<Tensor>"));
        assert!(matches!(v, SigDiffVerdict::Breaking { .. }));
    }

    #[test]
    fn output_type_change_breaking() {
        let v = diff(&base(), &sig("infer", &["Tensor", "u32"], "Tensor"));
        if let SigDiffVerdict::Breaking { reason } = v {
            assert!(reason.contains("output"));
        }
    }

    #[test]
    fn input_removal_breaking() {
        let v = diff(&base(), &sig("infer", &["Tensor"], "Result<Tensor>"));
        if let SigDiffVerdict::Breaking { reason } = v {
            assert!(reason.contains("removed"));
        }
    }

    #[test]
    fn rename_recognized() {
        let v = diff(
            &base(),
            &sig("predict", &["Tensor", "u32"], "Result<Tensor>"),
        );
        assert_eq!(v, SigDiffVerdict::Renamed);
    }

    #[test]
    fn empty_name_rejected() {
        let v = diff(&sig("", &[], "Result<()>"), &base());
        assert_eq!(v, SigDiffVerdict::EmptySignature);
    }

    #[test]
    fn no_inputs_works() {
        let a = sig("noop", &[], "()");
        let b = sig("noop", &[], "()");
        assert_eq!(diff(&a, &b), SigDiffVerdict::Identical);
    }

    #[test]
    fn additive_two_inputs() {
        let v = diff(
            &base(),
            &sig("infer", &["Tensor", "u32", "A", "B"], "Result<Tensor>"),
        );
        if let SigDiffVerdict::Additive { added_input_count } = v {
            assert_eq!(added_input_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let a = diff(&base(), &base());
        let b = diff(&base(), &base());
        assert_eq!(a, b);
    }
}
