//! # Contracts-Macros Proof Obligation Score
//!
//! Score how thoroughly each obligation is proven across 5 axes:
//! Spec, Falsification, Kani, Lean, Binding. Returns the per-
//! obligation score 0..=5 and the worst axis.
//!
//! Demonstrates the **CMM.58** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: provable_contracts pv score 5-axis scheme.
//!
//! Run with: cargo run --example contracts_macros_proof_obligation_score
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisStatus {
    pub spec: bool,
    pub falsification: bool,
    pub kani: bool,
    pub lean: bool,
    pub binding: bool,
}

#[derive(Debug, PartialEq)]
pub enum ScoreVerdict {
    Ok {
        score: u8,
        worst_axis: Option<&'static str>,
    },
    InvalidConfig,
}

pub fn score(status: AxisStatus) -> ScoreVerdict {
    let mut s: u8 = 0;
    if status.spec {
        s += 1;
    }
    if status.falsification {
        s += 1;
    }
    if status.kani {
        s += 1;
    }
    if status.lean {
        s += 1;
    }
    if status.binding {
        s += 1;
    }
    let worst_axis = if !status.spec {
        Some("spec")
    } else if !status.falsification {
        Some("falsification")
    } else if !status.kani {
        Some("kani")
    } else if !status.lean {
        Some("lean")
    } else if !status.binding {
        Some("binding")
    } else {
        None
    };
    ScoreVerdict::Ok {
        score: s,
        worst_axis,
    }
}

fn full() -> AxisStatus {
    AxisStatus {
        spec: true,
        falsification: true,
        kani: true,
        lean: true,
        binding: true,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_proof_obligation_score")?;

    println!("perfect: {:?}", score(full()));
    println!(
        "missing kani: {:?}",
        score(AxisStatus {
            kani: false,
            ..full()
        })
    );
    println!(
        "missing all: {:?}",
        score(AxisStatus {
            spec: false,
            falsification: false,
            kani: false,
            lean: false,
            binding: false,
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perfect_score_5() {
        if let ScoreVerdict::Ok { score, worst_axis } = score(full()) {
            assert_eq!(score, 5);
            assert!(worst_axis.is_none());
        }
    }

    #[test]
    fn missing_spec_first_worst() {
        let v = score(AxisStatus {
            spec: false,
            ..full()
        });
        if let ScoreVerdict::Ok { score, worst_axis } = v {
            assert_eq!(score, 4);
            assert_eq!(worst_axis, Some("spec"));
        }
    }

    #[test]
    fn missing_kani() {
        let v = score(AxisStatus {
            kani: false,
            ..full()
        });
        if let ScoreVerdict::Ok { worst_axis, .. } = v {
            assert_eq!(worst_axis, Some("kani"));
        }
    }

    #[test]
    fn missing_lean() {
        let v = score(AxisStatus {
            lean: false,
            ..full()
        });
        if let ScoreVerdict::Ok { worst_axis, .. } = v {
            assert_eq!(worst_axis, Some("lean"));
        }
    }

    #[test]
    fn missing_binding() {
        let v = score(AxisStatus {
            binding: false,
            ..full()
        });
        if let ScoreVerdict::Ok { worst_axis, .. } = v {
            assert_eq!(worst_axis, Some("binding"));
        }
    }

    #[test]
    fn empty_status_score_zero() {
        let v = score(AxisStatus {
            spec: false,
            falsification: false,
            kani: false,
            lean: false,
            binding: false,
        });
        if let ScoreVerdict::Ok { score, .. } = v {
            assert_eq!(score, 0);
        }
    }

    #[test]
    fn three_of_five() {
        let v = score(AxisStatus {
            spec: true,
            falsification: true,
            kani: true,
            lean: false,
            binding: false,
        });
        if let ScoreVerdict::Ok { score, worst_axis } = v {
            assert_eq!(score, 3);
            assert_eq!(worst_axis, Some("lean"));
        }
    }

    #[test]
    fn worst_axis_priority_order() {
        // When multiple missing, spec is reported first.
        let v = score(AxisStatus {
            spec: false,
            falsification: false,
            kani: false,
            lean: false,
            binding: true,
        });
        if let ScoreVerdict::Ok { worst_axis, .. } = v {
            assert_eq!(worst_axis, Some("spec"));
        }
    }

    #[test]
    fn deterministic() {
        let s = full();
        let a = score(s);
        let b = score(s);
        assert_eq!(a, b);
    }

    #[test]
    fn one_axis_pass() {
        let v = score(AxisStatus {
            spec: true,
            falsification: false,
            kani: false,
            lean: false,
            binding: false,
        });
        if let ScoreVerdict::Ok { score, .. } = v {
            assert_eq!(score, 1);
        }
    }
}
