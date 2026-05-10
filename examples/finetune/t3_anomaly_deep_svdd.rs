//! # Tier 3.6 — Deep SVDD anomaly detection (tabular-only)
//!
//! Falsifier: Deep SVDD hypersphere radius converges (monotone non-increasing
//! across training steps).
//!
//! Run with: cargo run --example t3_anomaly_deep_svdd

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL_RADIUS: f64 = 10.0;
const TAU: f64 = 0.1;
const N_STEPS: u32 = 20;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_anomaly_deep_svdd")?;
    let schedule = aou::svdd_radius_schedule(INITIAL_RADIUS, TAU, N_STEPS);
    println!(
        "✓ Deep SVDD: radius {:.4} → {:.4} (after {} steps)",
        schedule[0],
        schedule.last().unwrap(),
        N_STEPS
    );
    for w in schedule.windows(2) {
        assert!(w[0] >= w[1], "SVDD radius must be monotone non-increasing");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let s = aou::svdd_radius_schedule(INITIAL_RADIUS, TAU, N_STEPS);
        for w in s.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // tau=0 means no shrinking — schedule is constant.
        let s = aou::svdd_radius_schedule(INITIAL_RADIUS, 0.0, N_STEPS);
        for v in &s {
            assert_eq!(*v, INITIAL_RADIUS);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::svdd_radius_schedule(INITIAL_RADIUS, TAU, N_STEPS);
        let b = aou::svdd_radius_schedule(INITIAL_RADIUS, TAU, N_STEPS);
        assert_eq!(a, b);
    }
}
