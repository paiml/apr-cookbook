//! # apr debug --breakpoint — Step Planner
//!
//! `apr debug --breakpoint <SPEC>` accepts step expressions: a literal
//! step (`100`), an interval (`every:50`), or a list (`100,500,1000`).
//! Constraints: steps in [1, total_steps]; interval ≥ 1; list elements
//! sorted + unique. This recipe builds the planner.
//!
//! Demonstrates the **DBG.5** recipe for PMAT-117 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DBG-001 + GDB breakpoint syntax
//!
//! Run with: cargo run --example cli_debug_breakpoint_planner
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BreakpointPlan {
    Single(u32),
    Interval(u32),
    List(Vec<u32>),
}

#[derive(Debug, PartialEq)]
pub enum PlanVerdict {
    Ok(BreakpointPlan),
    InvalidSpec,
    StepZero,
    OutOfRange { step: u32, max: u32 },
    EmptyList,
    UnsortedList,
    DuplicateInList,
}

pub fn parse(spec: &str, total_steps: u32) -> PlanVerdict {
    if let Some(rest) = spec.strip_prefix("every:") {
        let interval: u32 = match rest.parse() {
            Ok(n) => n,
            Err(_) => return PlanVerdict::InvalidSpec,
        };
        if interval == 0 {
            return PlanVerdict::StepZero;
        }
        return PlanVerdict::Ok(BreakpointPlan::Interval(interval));
    }
    if spec.contains(',') {
        let mut nums: Vec<u32> = Vec::new();
        for part in spec.split(',') {
            let n: u32 = match part.trim().parse() {
                Ok(n) => n,
                Err(_) => return PlanVerdict::InvalidSpec,
            };
            if n == 0 {
                return PlanVerdict::StepZero;
            }
            if n > total_steps {
                return PlanVerdict::OutOfRange {
                    step: n,
                    max: total_steps,
                };
            }
            nums.push(n);
        }
        if nums.is_empty() {
            return PlanVerdict::EmptyList;
        }
        if nums.windows(2).any(|w| w[0] >= w[1]) {
            return if nums.windows(2).any(|w| w[0] == w[1]) {
                PlanVerdict::DuplicateInList
            } else {
                PlanVerdict::UnsortedList
            };
        }
        return PlanVerdict::Ok(BreakpointPlan::List(nums));
    }
    let n: u32 = match spec.parse() {
        Ok(n) => n,
        Err(_) => return PlanVerdict::InvalidSpec,
    };
    if n == 0 {
        return PlanVerdict::StepZero;
    }
    if n > total_steps {
        return PlanVerdict::OutOfRange {
            step: n,
            max: total_steps,
        };
    }
    PlanVerdict::Ok(BreakpointPlan::Single(n))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_breakpoint_planner")?;

    let cases = [
        ("100", 1000u32),
        ("every:50", 1000),
        ("100,500,1000", 1000),
        ("every:0", 1000),
        ("100,500,500", 1000),
        ("500,100", 1000),
    ];
    for (spec, total) in cases {
        println!("{spec:>14}  →  {:?}", parse(spec, total));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_step_parses() {
        assert_eq!(
            parse("100", 1000),
            PlanVerdict::Ok(BreakpointPlan::Single(100))
        );
    }

    #[test]
    fn interval_parses() {
        assert_eq!(
            parse("every:50", 1000),
            PlanVerdict::Ok(BreakpointPlan::Interval(50))
        );
    }

    #[test]
    fn list_parses_sorted() {
        let v = parse("100,500,1000", 1000);
        assert_eq!(
            v,
            PlanVerdict::Ok(BreakpointPlan::List(vec![100, 500, 1000]))
        );
    }

    #[test]
    fn step_zero_rejected() {
        assert_eq!(parse("0", 1000), PlanVerdict::StepZero);
        assert_eq!(parse("every:0", 1000), PlanVerdict::StepZero);
    }

    #[test]
    fn out_of_range_rejected() {
        let v = parse("2000", 1000);
        assert!(matches!(v, PlanVerdict::OutOfRange { .. }));
    }

    #[test]
    fn unsorted_list_rejected() {
        assert_eq!(parse("500,100", 1000), PlanVerdict::UnsortedList);
    }

    #[test]
    fn duplicate_in_list_rejected() {
        assert_eq!(parse("100,500,500", 1000), PlanVerdict::DuplicateInList);
    }

    #[test]
    fn invalid_spec_rejected() {
        assert_eq!(parse("not_a_number", 1000), PlanVerdict::InvalidSpec);
        assert_eq!(parse("every:foo", 1000), PlanVerdict::InvalidSpec);
    }

    #[test]
    fn whitespace_in_list_tolerated() {
        let v = parse("100, 500, 1000", 1000);
        assert_eq!(
            v,
            PlanVerdict::Ok(BreakpointPlan::List(vec![100, 500, 1000]))
        );
    }
}
