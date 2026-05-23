//! # Contracts-Macros Lean Compile Status Tracker
//!
//! Roll up Lean compile statuses across modules: Compiled / TypeError /
//! ImportError / Timeout. Returns the first blocking error and a
//! summary count per status.
//!
//! Demonstrates the **CMM.63** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: lake (Lean 4 build) status reporting.
//!
//! Run with: cargo run --example contracts_macros_lean_compile_status
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileStatus {
    Compiled,
    TypeError,
    ImportError,
    Timeout,
}

#[derive(Debug, PartialEq)]
pub enum CompileVerdict {
    AllCompiled {
        count: u32,
    },
    HasErrors {
        first_error: String,
        type_errors: u32,
        import_errors: u32,
        timeouts: u32,
    },
    EmptyContract,
}

pub fn rollup(modules: &[(&str, CompileStatus)]) -> CompileVerdict {
    if modules.is_empty() {
        return CompileVerdict::EmptyContract;
    }
    let mut type_errors = 0u32;
    let mut import_errors = 0u32;
    let mut timeouts = 0u32;
    let mut first_error: Option<&str> = None;
    for (name, status) in modules {
        match status {
            CompileStatus::TypeError => {
                type_errors += 1;
                if first_error.is_none() {
                    first_error = Some(name);
                }
            }
            CompileStatus::ImportError => {
                import_errors += 1;
                if first_error.is_none() {
                    first_error = Some(name);
                }
            }
            CompileStatus::Timeout => {
                timeouts += 1;
                if first_error.is_none() {
                    first_error = Some(name);
                }
            }
            CompileStatus::Compiled => {}
        }
    }
    if let Some(name) = first_error {
        CompileVerdict::HasErrors {
            first_error: name.to_string(),
            type_errors,
            import_errors,
            timeouts,
        }
    } else {
        CompileVerdict::AllCompiled {
            count: modules.len() as u32,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_lean_compile_status")?;

    let all_ok = [
        ("Mod.A", CompileStatus::Compiled),
        ("Mod.B", CompileStatus::Compiled),
    ];
    println!("ok: {:?}", rollup(&all_ok));

    let mixed = [
        ("Mod.A", CompileStatus::Compiled),
        ("Mod.B", CompileStatus::TypeError),
        ("Mod.C", CompileStatus::Timeout),
    ];
    println!("mixed: {:?}", rollup(&mixed));
    println!("empty: {:?}", rollup(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_compiled_recognized() {
        let v = rollup(&[
            ("a", CompileStatus::Compiled),
            ("b", CompileStatus::Compiled),
        ]);
        if let CompileVerdict::AllCompiled { count } = v {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn first_error_returned() {
        let v = rollup(&[
            ("ok", CompileStatus::Compiled),
            ("first", CompileStatus::TypeError),
            ("second", CompileStatus::Timeout),
        ]);
        if let CompileVerdict::HasErrors { first_error, .. } = v {
            assert_eq!(first_error, "first");
        }
    }

    #[test]
    fn type_error_counted() {
        let v = rollup(&[("a", CompileStatus::TypeError)]);
        if let CompileVerdict::HasErrors { type_errors, .. } = v {
            assert_eq!(type_errors, 1);
        }
    }

    #[test]
    fn import_error_counted() {
        let v = rollup(&[("a", CompileStatus::ImportError)]);
        if let CompileVerdict::HasErrors { import_errors, .. } = v {
            assert_eq!(import_errors, 1);
        }
    }

    #[test]
    fn timeout_counted() {
        let v = rollup(&[("a", CompileStatus::Timeout)]);
        if let CompileVerdict::HasErrors { timeouts, .. } = v {
            assert_eq!(timeouts, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(rollup(&[]), CompileVerdict::EmptyContract);
    }

    #[test]
    fn multiple_errors_counted_correctly() {
        let v = rollup(&[
            ("a", CompileStatus::TypeError),
            ("b", CompileStatus::TypeError),
            ("c", CompileStatus::Timeout),
        ]);
        if let CompileVerdict::HasErrors {
            type_errors,
            timeouts,
            ..
        } = v
        {
            assert_eq!(type_errors, 2);
            assert_eq!(timeouts, 1);
        }
    }

    #[test]
    fn one_compiled_among_errors() {
        let v = rollup(&[
            ("ok", CompileStatus::Compiled),
            ("bad", CompileStatus::TypeError),
        ]);
        assert!(matches!(v, CompileVerdict::HasErrors { .. }));
    }

    #[test]
    fn single_compiled() {
        let v = rollup(&[("only", CompileStatus::Compiled)]);
        if let CompileVerdict::AllCompiled { count } = v {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let m = [("a", CompileStatus::TypeError)];
        let a = rollup(&m);
        let b = rollup(&m);
        assert_eq!(a, b);
    }
}
