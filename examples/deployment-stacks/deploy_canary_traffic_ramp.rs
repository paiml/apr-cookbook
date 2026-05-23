//! # Deployment Canary Traffic Ramp
//!
//! Canary releases gradually shift traffic from baseline to canary in
//! steps (e.g., 1% → 10% → 50% → 100%). Each step gates on health
//! checks. This recipe builds the step planner + per-step pause
//! duration calculator (longer at higher pcts to detect issues at
//! scale).
//!
//! Demonstrates the **DEPLOY.17** recipe for PMAT-130 (deployment-stacks coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Continuous Delivery (Humble & Farley, 2010) §10.
//!
//! Run with: cargo run --example deploy_canary_traffic_ramp
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RampVerdict {
    Ok(Vec<RampStep>),
    InvalidInitial,
    InvalidFinal,
    ZeroSteps,
    NonMonotonicSteps,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct RampStep {
    pub traffic_pct: u32,
    pub pause_minutes: u32,
}

pub fn build_ramp(start_pct: u32, end_pct: u32, num_steps: u32) -> RampVerdict {
    if start_pct == 0 || start_pct > 100 {
        return RampVerdict::InvalidInitial;
    }
    if end_pct == 0 || end_pct > 100 || end_pct <= start_pct {
        return RampVerdict::InvalidFinal;
    }
    if num_steps == 0 {
        return RampVerdict::ZeroSteps;
    }
    let span = end_pct - start_pct;
    if num_steps > span {
        return RampVerdict::NonMonotonicSteps;
    }
    let step_size = span / num_steps;
    let mut steps = Vec::with_capacity(num_steps as usize);
    for i in 0..num_steps {
        let pct = if i + 1 == num_steps {
            end_pct
        } else {
            start_pct + step_size * (i + 1)
        };
        steps.push(RampStep {
            traffic_pct: pct,
            pause_minutes: pause_for_pct(pct),
        });
    }
    RampVerdict::Ok(steps)
}

pub fn pause_for_pct(pct: u32) -> u32 {
    // Higher traffic = longer observation window before next step.
    if pct < 5 {
        5
    } else if pct < 25 {
        15
    } else if pct < 75 {
        30
    } else {
        60
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("deploy_canary_traffic_ramp")?;

    println!("1→100 in 4 steps: {:?}", build_ramp(1, 100, 4));
    println!("invalid: {:?}", build_ramp(50, 30, 3));
    println!("zero steps: {:?}", build_ramp(1, 100, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_4_step_ramp() {
        if let RampVerdict::Ok(steps) = build_ramp(1, 100, 4) {
            assert_eq!(steps.len(), 4);
            assert!(steps[0].traffic_pct < steps[3].traffic_pct);
            assert_eq!(steps[3].traffic_pct, 100);
        }
    }

    #[test]
    fn final_step_reaches_target() {
        if let RampVerdict::Ok(steps) = build_ramp(5, 75, 3) {
            assert_eq!(steps.last().unwrap().traffic_pct, 75);
        }
    }

    #[test]
    fn pause_grows_with_pct() {
        assert!(pause_for_pct(1) < pause_for_pct(50));
        assert!(pause_for_pct(50) < pause_for_pct(99));
    }

    #[test]
    fn start_zero_invalid() {
        assert_eq!(build_ramp(0, 50, 2), RampVerdict::InvalidInitial);
    }

    #[test]
    fn start_over_100_invalid() {
        assert_eq!(build_ramp(101, 200, 2), RampVerdict::InvalidInitial);
    }

    #[test]
    fn end_at_or_below_start_invalid() {
        assert_eq!(build_ramp(50, 30, 2), RampVerdict::InvalidFinal);
        assert_eq!(build_ramp(50, 50, 2), RampVerdict::InvalidFinal);
    }

    #[test]
    fn zero_steps_rejected() {
        assert_eq!(build_ramp(1, 100, 0), RampVerdict::ZeroSteps);
    }

    #[test]
    fn too_many_steps_for_span_rejected() {
        // span = 5, but want 100 steps.
        let v = build_ramp(1, 6, 100);
        assert_eq!(v, RampVerdict::NonMonotonicSteps);
    }

    #[test]
    fn step_pcts_monotonically_increasing() {
        if let RampVerdict::Ok(steps) = build_ramp(1, 100, 5) {
            for w in steps.windows(2) {
                assert!(w[0].traffic_pct <= w[1].traffic_pct);
            }
        }
    }

    #[test]
    fn pause_tier_at_5_pct_boundary() {
        // < 5% gets shortest pause.
        assert_eq!(pause_for_pct(4), 5);
        assert_eq!(pause_for_pct(5), 15);
    }
}
