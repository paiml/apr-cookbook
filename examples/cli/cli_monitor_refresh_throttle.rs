//! # apr monitor — `--refresh-ms` Throttle
//!
//! `apr monitor --refresh-ms <N>` controls TUI redraw cadence. Below 50ms
//! the redraw outpaces the underlying training-step write rate (typical
//! 200 ms+) and burns CPU. Above 5000ms operators perceive the TUI as
//! "stuck". This recipe builds the throttle validator and asserts the
//! sane band.
//!
//! Demonstrates the **MONITOR.4** recipe for PMAT-101 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MONITOR-001 + TUI redraw best practices
//!
//! Run with: cargo run --example cli_monitor_refresh_throttle
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RefreshVerdict {
    Ok,
    TooFast { observed: u32, recommended_min: u32 },
    TooSlow { observed: u32, recommended_max: u32 },
}

const MIN_MS: u32 = 50;
const MAX_MS: u32 = 5000;

pub fn validate_refresh(ms: u32) -> RefreshVerdict {
    if ms < MIN_MS {
        return RefreshVerdict::TooFast {
            observed: ms,
            recommended_min: MIN_MS,
        };
    }
    if ms > MAX_MS {
        return RefreshVerdict::TooSlow {
            observed: ms,
            recommended_max: MAX_MS,
        };
    }
    RefreshVerdict::Ok
}

pub fn redraws_per_step(refresh_ms: u32, step_duration_ms: u32) -> f64 {
    if refresh_ms == 0 {
        return f64::INFINITY;
    }
    f64::from(step_duration_ms) / f64::from(refresh_ms)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_refresh_throttle")?;

    for ms in [10u32, 50, 100, 500, 5000, 6000] {
        let v = validate_refresh(ms);
        let ratio = redraws_per_step(ms, 200);
        println!("refresh={ms:>5}ms  →  {v:?}  (~{ratio:.2} redraws per 200ms step)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_500ms_is_ok() {
        // The CLI default — must be valid.
        assert_eq!(validate_refresh(500), RefreshVerdict::Ok);
    }

    #[test]
    fn boundary_at_min_passes() {
        // Conservative-pass at the minimum.
        assert_eq!(validate_refresh(50), RefreshVerdict::Ok);
    }

    #[test]
    fn boundary_at_max_passes() {
        assert_eq!(validate_refresh(5000), RefreshVerdict::Ok);
    }

    #[test]
    fn below_min_rejected() {
        assert!(matches!(
            validate_refresh(10),
            RefreshVerdict::TooFast { .. }
        ));
        assert!(matches!(
            validate_refresh(0),
            RefreshVerdict::TooFast { .. }
        ));
    }

    #[test]
    fn above_max_rejected() {
        assert!(matches!(
            validate_refresh(6000),
            RefreshVerdict::TooSlow { .. }
        ));
    }

    #[test]
    fn redraws_per_step_sane_for_default() {
        // 500ms refresh against 200ms step → 0.4 redraws per step (every other step).
        let r = redraws_per_step(500, 200);
        assert!((r - 0.4).abs() < 1e-9);
    }

    #[test]
    fn redraws_per_step_zero_refresh_returns_infinity() {
        // Avoid divide-by-zero — return +inf so caller sees the pathological case.
        assert!(redraws_per_step(0, 200).is_infinite());
    }
}
