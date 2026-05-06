//! # Contracts-Macros Dispatch Fallback
//!
//! When primary equation's preconditions don't hold, dispatch to a
//! fallback equation. Returns which one was selected and why.
//!
//! Demonstrates the **CMM.10** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Eiffel-style preconditions + multi-equation dispatch.
//!
//! Run with: cargo run --example contracts_macros_dispatch_fallback
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Primary {
        name: String,
    },
    Fallback {
        primary_name: String,
        fallback_name: String,
    },
    AllFailed,
    EmptyDispatch,
}

#[derive(Debug, Clone)]
pub struct EqCandidate {
    pub name: String,
    pub precond_ok: bool,
}

pub fn dispatch(candidates: &[EqCandidate]) -> DispatchVerdict {
    if candidates.is_empty() {
        return DispatchVerdict::EmptyDispatch;
    }
    if candidates[0].precond_ok {
        return DispatchVerdict::Primary {
            name: candidates[0].name.clone(),
        };
    }
    for c in candidates.iter().skip(1) {
        if c.precond_ok {
            return DispatchVerdict::Fallback {
                primary_name: candidates[0].name.clone(),
                fallback_name: c.name.clone(),
            };
        }
    }
    DispatchVerdict::AllFailed
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_dispatch_fallback")?;

    let cs = vec![
        EqCandidate {
            name: "fast_path".to_string(),
            precond_ok: true,
        },
        EqCandidate {
            name: "slow_path".to_string(),
            precond_ok: true,
        },
    ];
    println!("primary: {:?}", dispatch(&cs));

    let cs2 = vec![
        EqCandidate {
            name: "fast_path".to_string(),
            precond_ok: false,
        },
        EqCandidate {
            name: "slow_path".to_string(),
            precond_ok: true,
        },
    ];
    println!("fallback: {:?}", dispatch(&cs2));

    let cs3 = vec![EqCandidate {
        name: "x".to_string(),
        precond_ok: false,
    }];
    println!("all failed: {:?}", dispatch(&cs3));
    println!("empty: {:?}", dispatch(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ec(name: &str, ok: bool) -> EqCandidate {
        EqCandidate {
            name: name.to_string(),
            precond_ok: ok,
        }
    }

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn primary_when_first_ok() {
        let v = dispatch(&[ec("a", true), ec("b", true)]);
        if let DispatchVerdict::Primary { name } = v {
            assert_eq!(name, "a");
        }
    }

    #[test]
    fn fallback_when_primary_fails() {
        let v = dispatch(&[ec("a", false), ec("b", true)]);
        if let DispatchVerdict::Fallback {
            primary_name,
            fallback_name,
        } = v
        {
            assert_eq!(primary_name, "a");
            assert_eq!(fallback_name, "b");
        }
    }

    #[test]
    fn all_failed_signaled() {
        let v = dispatch(&[ec("a", false), ec("b", false)]);
        assert_eq!(v, DispatchVerdict::AllFailed);
    }

    #[test]
    fn empty_dispatch() {
        assert_eq!(dispatch(&[]), DispatchVerdict::EmptyDispatch);
    }

    #[test]
    fn first_ok_skips_fallback() {
        // Even if later candidates exist, primary wins.
        let v = dispatch(&[ec("a", true), ec("b", false)]);
        if let DispatchVerdict::Primary { name } = v {
            assert_eq!(name, "a");
        }
    }

    #[test]
    fn second_fallback_picked() {
        let v = dispatch(&[ec("a", false), ec("b", false), ec("c", true)]);
        if let DispatchVerdict::Fallback { fallback_name, .. } = v {
            assert_eq!(fallback_name, "c");
        }
    }

    #[test]
    fn first_fallback_wins_over_later() {
        let v = dispatch(&[ec("a", false), ec("b", true), ec("c", true)]);
        if let DispatchVerdict::Fallback { fallback_name, .. } = v {
            assert_eq!(fallback_name, "b");
        }
    }

    #[test]
    fn single_ok_primary() {
        let v = dispatch(&[ec("only", true)]);
        if let DispatchVerdict::Primary { name } = v {
            assert_eq!(name, "only");
        }
    }

    #[test]
    fn single_failed_all_failed() {
        let v = dispatch(&[ec("only", false)]);
        assert_eq!(v, DispatchVerdict::AllFailed);
    }

    #[test]
    fn deterministic() {
        let cs = [ec("a", false), ec("b", true)];
        let a = dispatch(&cs);
        let b = dispatch(&cs);
        assert_eq!(a, b);
    }
}
