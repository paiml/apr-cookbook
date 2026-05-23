//! # Deployment Blue-Green Cutover Validator
//!
//! Blue-green deployment: blue = current production, green = new
//! version. Cutover requires: green pods all healthy, green smoke-test
//! passes, blue still up for instant rollback. This recipe codifies
//! the precondition gate.
//!
//! Demonstrates the **DEPLOY.16** recipe for PMAT-130 (deployment-stacks coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Fowler, M. (2010). BlueGreenDeployment.
//!
//! Run with: cargo run --example deploy_blue_green_cutover
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct StackHealth {
    pub healthy_pods: u32,
    pub total_pods: u32,
    pub smoke_test_passed: bool,
    pub readiness_probe_pct: f64,
}

#[derive(Debug, PartialEq)]
pub enum CutoverVerdict {
    Ok,
    GreenNotReady { healthy: u32, total: u32 },
    GreenSmokeFailed,
    GreenReadinessLow { pct: f64, required: f64 },
    BlueNotAvailable,
    InvalidConfig,
}

const REQUIRED_READINESS_PCT: f64 = 95.0;

pub fn validate(blue: StackHealth, green: StackHealth) -> CutoverVerdict {
    if green.total_pods == 0 {
        return CutoverVerdict::InvalidConfig;
    }
    // Blue must remain available for rollback.
    if blue.healthy_pods == 0 || blue.total_pods == 0 {
        return CutoverVerdict::BlueNotAvailable;
    }
    if green.healthy_pods != green.total_pods {
        return CutoverVerdict::GreenNotReady {
            healthy: green.healthy_pods,
            total: green.total_pods,
        };
    }
    if !green.smoke_test_passed {
        return CutoverVerdict::GreenSmokeFailed;
    }
    if green.readiness_probe_pct < REQUIRED_READINESS_PCT {
        return CutoverVerdict::GreenReadinessLow {
            pct: green.readiness_probe_pct,
            required: REQUIRED_READINESS_PCT,
        };
    }
    CutoverVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("deploy_blue_green_cutover")?;

    let blue_healthy = StackHealth {
        healthy_pods: 5,
        total_pods: 5,
        smoke_test_passed: true,
        readiness_probe_pct: 100.0,
    };
    let green_ok = StackHealth {
        healthy_pods: 5,
        total_pods: 5,
        smoke_test_passed: true,
        readiness_probe_pct: 99.0,
    };
    println!("ok: {:?}", validate(blue_healthy, green_ok));

    let mut green_partial = green_ok;
    green_partial.healthy_pods = 3;
    println!("partial green: {:?}", validate(blue_healthy, green_partial));

    let mut green_smoke_fail = green_ok;
    green_smoke_fail.smoke_test_passed = false;
    println!("smoke fail: {:?}", validate(blue_healthy, green_smoke_fail));

    let blue_dead = StackHealth {
        healthy_pods: 0,
        ..blue_healthy
    };
    println!("blue dead: {:?}", validate(blue_dead, green_ok));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> StackHealth {
        StackHealth {
            healthy_pods: 5,
            total_pods: 5,
            smoke_test_passed: true,
            readiness_probe_pct: 99.0,
        }
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fully_healthy_pair_passes() {
        assert_eq!(validate(good(), good()), CutoverVerdict::Ok);
    }

    #[test]
    fn green_not_fully_ready_rejected() {
        let mut green = good();
        green.healthy_pods = 3;
        let v = validate(good(), green);
        assert!(matches!(v, CutoverVerdict::GreenNotReady { .. }));
    }

    #[test]
    fn green_smoke_fail_rejected() {
        let mut green = good();
        green.smoke_test_passed = false;
        assert_eq!(validate(good(), green), CutoverVerdict::GreenSmokeFailed);
    }

    #[test]
    fn green_readiness_too_low_rejected() {
        let mut green = good();
        green.readiness_probe_pct = 50.0;
        let v = validate(good(), green);
        assert!(matches!(v, CutoverVerdict::GreenReadinessLow { .. }));
    }

    #[test]
    fn blue_not_available_rejected() {
        let mut blue = good();
        blue.healthy_pods = 0;
        assert_eq!(validate(blue, good()), CutoverVerdict::BlueNotAvailable);
    }

    #[test]
    fn empty_green_invalid_config() {
        let mut green = good();
        green.total_pods = 0;
        green.healthy_pods = 0;
        assert_eq!(validate(good(), green), CutoverVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_at_required_readiness_passes() {
        let mut green = good();
        green.readiness_probe_pct = REQUIRED_READINESS_PCT;
        assert_eq!(validate(good(), green), CutoverVerdict::Ok);
    }

    #[test]
    fn just_below_readiness_rejected() {
        let mut green = good();
        green.readiness_probe_pct = REQUIRED_READINESS_PCT - 0.01;
        let v = validate(good(), green);
        assert!(matches!(v, CutoverVerdict::GreenReadinessLow { .. }));
    }
}
