//! # apr tune --scheduler — Pruning Scheduler Dispatcher
//!
//! `apr tune --scheduler <S>` accepts {asha, median, none}. ASHA
//! (Asynchronous Successive Halving) is the modern default — promotes
//! top fraction every rung. Median pruner stops trials below median at
//! each step. None disables pruning. Dispatch rules: any scheduler
//! requires reporting intermediate values; `none` is fine for
//! short-budget jobs.
//!
//! Demonstrates the **TUNE.8** recipe for PMAT-111 (apr tune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender HPO-001 + Li et al. 2018 (ASHA)
//!
//! Run with: cargo run --example cli_tune_scheduler_dispatcher
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduler {
    None,
    Median,
    Asha,
}

impl Scheduler {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Scheduler::None),
            "median" => Some(Scheduler::Median),
            "asha" => Some(Scheduler::Asha),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok,
    SchedulerRequiresIntermediateValues,
    SchedulerOverkillForBudget,
}

pub fn dispatch(
    scheduler: Scheduler,
    reports_intermediate: bool,
    num_trials: u32,
) -> DispatchVerdict {
    if scheduler == Scheduler::None {
        return DispatchVerdict::Ok;
    }
    if !reports_intermediate {
        return DispatchVerdict::SchedulerRequiresIntermediateValues;
    }
    if num_trials < 5 {
        // Pruning needs enough trials to learn from; skip overhead for tiny jobs.
        return DispatchVerdict::SchedulerOverkillForBudget;
    }
    DispatchVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tune_scheduler_dispatcher")?;

    let cases = [
        ("none/no-intermediate", Scheduler::None, false, 100),
        ("asha/intermediate/100", Scheduler::Asha, true, 100),
        ("median/no-intermediate", Scheduler::Median, false, 100),
        ("asha/intermediate/3", Scheduler::Asha, true, 3),
    ];
    for (label, s, reports, n) in cases {
        println!("{label:>26}  →  {:?}", dispatch(s, reports, n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn none_always_ok() {
        // Disabling pruning has no preconditions.
        assert_eq!(dispatch(Scheduler::None, false, 1), DispatchVerdict::Ok);
        assert_eq!(dispatch(Scheduler::None, true, 1000), DispatchVerdict::Ok);
    }

    #[test]
    fn scheduler_without_intermediate_rejected() {
        // ASHA + median both need stepwise validation reports.
        assert_eq!(
            dispatch(Scheduler::Asha, false, 100),
            DispatchVerdict::SchedulerRequiresIntermediateValues
        );
        assert_eq!(
            dispatch(Scheduler::Median, false, 100),
            DispatchVerdict::SchedulerRequiresIntermediateValues
        );
    }

    #[test]
    fn scheduler_with_tiny_budget_rejected() {
        // < 5 trials → not worth scheduler overhead.
        assert_eq!(
            dispatch(Scheduler::Asha, true, 4),
            DispatchVerdict::SchedulerOverkillForBudget
        );
    }

    #[test]
    fn scheduler_with_adequate_budget_ok() {
        assert_eq!(dispatch(Scheduler::Asha, true, 100), DispatchVerdict::Ok);
        assert_eq!(dispatch(Scheduler::Median, true, 50), DispatchVerdict::Ok);
    }

    #[test]
    fn boundary_at_5_trials_passes() {
        assert_eq!(dispatch(Scheduler::Asha, true, 5), DispatchVerdict::Ok);
    }

    #[test]
    fn known_schedulers_round_trip() {
        for s in ["none", "median", "asha"] {
            assert!(Scheduler::from_str_strict(s).is_some());
        }
        assert!(Scheduler::from_str_strict("hyperband").is_none());
    }
}
