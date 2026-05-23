//! # apr qa --tier — Per-Tier Gate Aggregator
//!
//! `apr qa` runs gates organized into tiers (lint, unit, integration,
//! e2e, perf). Aggregation rule: highest-severity tier wins, but
//! lower-tier failures still surface in the report. This recipe builds
//! the aggregator + tier-priority ordering.
//!
//! Demonstrates the **QA.4** recipe for PMAT-121 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QA-001 + Toyota Way (jidoka — built-in quality)
//!
//! Run with: cargo run --example cli_qa_tier_aggregator
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Lint,
    Unit,
    Integration,
    E2e,
    Perf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierResult {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, PartialEq)]
pub enum AggregateVerdict {
    AllPass,
    HighestFailedTier(Tier),
    AllSkipped,
}

pub fn aggregate(results: &[(Tier, TierResult)]) -> AggregateVerdict {
    if results.is_empty() {
        return AggregateVerdict::AllSkipped;
    }
    if results.iter().all(|(_, r)| *r == TierResult::Skip) {
        return AggregateVerdict::AllSkipped;
    }
    let mut highest_fail: Option<Tier> = None;
    for (tier, result) in results {
        if *result == TierResult::Fail {
            highest_fail = Some(match highest_fail {
                Some(h) if h >= *tier => h,
                _ => *tier,
            });
        }
    }
    match highest_fail {
        Some(t) => AggregateVerdict::HighestFailedTier(t),
        None => AggregateVerdict::AllPass,
    }
}

pub fn tier_priority(tier: Tier) -> u8 {
    match tier {
        Tier::Lint => 1,
        Tier::Unit => 2,
        Tier::Integration => 3,
        Tier::E2e => 4,
        Tier::Perf => 5,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_tier_aggregator")?;

    let all_pass = vec![
        (Tier::Lint, TierResult::Pass),
        (Tier::Unit, TierResult::Pass),
        (Tier::E2e, TierResult::Pass),
    ];
    println!("all pass: {:?}", aggregate(&all_pass));

    let mixed = vec![
        (Tier::Lint, TierResult::Pass),
        (Tier::Unit, TierResult::Fail),
        (Tier::Integration, TierResult::Pass),
        (Tier::E2e, TierResult::Fail),
        (Tier::Perf, TierResult::Pass),
    ];
    println!("mixed:    {:?}", aggregate(&mixed));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_results_skipped() {
        assert_eq!(aggregate(&[]), AggregateVerdict::AllSkipped);
    }

    #[test]
    fn all_skipped_yields_skipped() {
        let results = vec![
            (Tier::Lint, TierResult::Skip),
            (Tier::Unit, TierResult::Skip),
        ];
        assert_eq!(aggregate(&results), AggregateVerdict::AllSkipped);
    }

    #[test]
    fn all_pass_aggregates_pass() {
        let results = vec![
            (Tier::Lint, TierResult::Pass),
            (Tier::Unit, TierResult::Pass),
        ];
        assert_eq!(aggregate(&results), AggregateVerdict::AllPass);
    }

    #[test]
    fn single_fail_returns_that_tier() {
        let results = vec![
            (Tier::Lint, TierResult::Pass),
            (Tier::Unit, TierResult::Fail),
        ];
        assert_eq!(
            aggregate(&results),
            AggregateVerdict::HighestFailedTier(Tier::Unit)
        );
    }

    #[test]
    fn highest_fail_wins_when_multiple() {
        let results = vec![
            (Tier::Unit, TierResult::Fail),
            (Tier::E2e, TierResult::Fail),
            (Tier::Lint, TierResult::Fail),
        ];
        assert_eq!(
            aggregate(&results),
            AggregateVerdict::HighestFailedTier(Tier::E2e)
        );
    }

    #[test]
    fn perf_fail_is_highest() {
        let results = vec![
            (Tier::Lint, TierResult::Fail),
            (Tier::Perf, TierResult::Fail),
        ];
        assert_eq!(
            aggregate(&results),
            AggregateVerdict::HighestFailedTier(Tier::Perf)
        );
    }

    #[test]
    fn skip_does_not_count_as_fail() {
        let results = vec![
            (Tier::Lint, TierResult::Skip),
            (Tier::Unit, TierResult::Pass),
        ];
        assert_eq!(aggregate(&results), AggregateVerdict::AllPass);
    }

    #[test]
    fn priority_ordering_is_strict() {
        assert!(tier_priority(Tier::Lint) < tier_priority(Tier::Unit));
        assert!(tier_priority(Tier::Unit) < tier_priority(Tier::Integration));
        assert!(tier_priority(Tier::Integration) < tier_priority(Tier::E2e));
        assert!(tier_priority(Tier::E2e) < tier_priority(Tier::Perf));
    }
}
