//! # Tier 4.9 — CPO contrastive preference (mistral family)
//!
//! Falsifier: CPO chosen-rejected margin grows monotonically over training.
//!
//! Run with: cargo run --example t4_cpo

use apr_cookbook::finetune::online_alt as oa;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL: f64 = 0.0;
const SLOPE: f64 = 0.01;
const N_STEPS: u32 = 50;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_cpo")?;
    let traj = oa::cpo_margin_trajectory(INITIAL, SLOPE, N_STEPS);
    println!(
        "✓ CPO margin: {} steps, {:.3} → {:.3}",
        N_STEPS,
        traj[0],
        traj.last().unwrap()
    );
    for w in traj.windows(2) {
        assert!(w[1] >= w[0]);
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
        let traj = oa::cpo_margin_trajectory(INITIAL, SLOPE, N_STEPS);
        for w in traj.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Negative slope — margin shrinks.
        let traj = oa::cpo_margin_trajectory(INITIAL, -0.01, N_STEPS);
        for w in traj.windows(2) {
            assert!(w[1] <= w[0]);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = oa::cpo_margin_trajectory(INITIAL, SLOPE, N_STEPS);
        let b = oa::cpo_margin_trajectory(INITIAL, SLOPE, N_STEPS);
        assert_eq!(a, b);
    }
}
