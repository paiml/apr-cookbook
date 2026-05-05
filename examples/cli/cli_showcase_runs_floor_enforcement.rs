//! # apr showcase — `--runs` Statistical Floor
//!
//! `apr showcase --runs <N>` controls the benchmark sample size. The spec
//! mandates **minimum 30 runs** for the t-distribution central-limit
//! approximation to be valid (used to compute confidence intervals on the
//! tok/s comparison vs llama-cpp/ollama). This recipe documents and
//! enforces the floor.
//!
//! Demonstrates the **SHOWCASE.5** recipe for PMAT-096 (apr showcase coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHOWCASE-003 + Student's t-distribution (n≥30 rule)
//!
//! Run with: cargo run --example cli_showcase_runs_floor_enforcement
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RunsVerdict {
    Ok,
    BelowMinimum { observed: u32, required: u32 },
    Excessive { observed: u32 },
}

const MIN_RUNS: u32 = 30;
const EXCESSIVE_RUNS: u32 = 10_000;

pub fn validate_runs(n: u32) -> RunsVerdict {
    if n < MIN_RUNS {
        return RunsVerdict::BelowMinimum {
            observed: n,
            required: MIN_RUNS,
        };
    }
    if n > EXCESSIVE_RUNS {
        // Refusing to run obviously-pointless benchmarks (10K+ runs takes hours
        // and the CLT effect tops out around 100).
        return RunsVerdict::Excessive { observed: n };
    }
    RunsVerdict::Ok
}

/// Total wall-clock estimate at ~10s per run × 4 baselines (rough).
pub fn estimate_wallclock_seconds(n: u32, baseline_count: u32) -> u64 {
    u64::from(n) * 10 * u64::from(baseline_count)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_showcase_runs_floor_enforcement")?;

    for n in [10u32, 29, 30, 100, 1_000, 50_000] {
        let v = validate_runs(n);
        let eta = estimate_wallclock_seconds(n, 2);
        println!("--runs {n:>5}  →  {v:?}  (est ~{eta}s for 2 baselines)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_30_passes() {
        // The CLI default; must be valid.
        assert_eq!(validate_runs(30), RunsVerdict::Ok);
    }

    #[test]
    fn below_30_rejected() {
        let v = validate_runs(29);
        assert!(matches!(
            v,
            RunsVerdict::BelowMinimum {
                observed: 29,
                required: 30
            }
        ));
    }

    #[test]
    fn zero_runs_rejected() {
        // Edge: zero is below the floor — must reject (not silently pass).
        let v = validate_runs(0);
        assert!(matches!(v, RunsVerdict::BelowMinimum { observed: 0, .. }));
    }

    #[test]
    fn excessive_runs_rejected() {
        // Way above 10K is almost certainly an operator typo.
        assert!(matches!(
            validate_runs(50_000),
            RunsVerdict::Excessive { .. }
        ));
    }

    #[test]
    fn boundary_at_exactly_10000_passes() {
        // Conservative-pass at the upper bound.
        assert_eq!(validate_runs(10_000), RunsVerdict::Ok);
    }

    #[test]
    fn wallclock_estimate_scales_linearly() {
        // 30 runs × 2 baselines × 10 s = 600 s.
        assert_eq!(estimate_wallclock_seconds(30, 2), 600);
        assert_eq!(estimate_wallclock_seconds(60, 2), 1200);
    }
}
