//! # Tier 2.6 — BAdam block-wise update (mistral family)
//!
//! Falsifier: BAdam block-wise update — per-block parameter mass conserved
//! across optimizer step (only the active block updates; others retain
//! L1 mass identically).
//!
//! Run with: cargo run --example t2_badam

use apr_cookbook::finetune::memory_optimizers as mem;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_params() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let params = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let update = vec![0.5; 8];
    let block_starts = vec![0, 2, 4, 6]; // 4 blocks of size 2
    (params, update, block_starts)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_badam")?;
    let (params, update, starts) = fixture_params();
    let active = 1;
    let after = mem::block_mass_after_update(&params, &update, &starts, active);
    let before_inactive_mass: Vec<f64> = (0..starts.len())
        .filter(|&b| b != active)
        .map(|b| {
            let s = starts[b];
            let e = starts.get(b + 1).copied().unwrap_or(params.len());
            mem::l1_mass(&params[s..e])
        })
        .collect();
    let after_inactive_mass: Vec<f64> = (0..starts.len())
        .filter(|&b| b != active)
        .map(|b| {
            let s = starts[b];
            let e = starts.get(b + 1).copied().unwrap_or(params.len());
            mem::l1_mass(&after[s..e])
        })
        .collect();
    println!(
        "✓ BAdam active=block {}: peak_mem_ratio={:.4}",
        active,
        mem::badam_peak_memory_ratio(starts.len() as u32)
    );
    for (b, a) in before_inactive_mass.iter().zip(after_inactive_mass.iter()) {
        assert!(
            (b - a).abs() < 1e-12,
            "BAdam must preserve inactive-block mass: {b} → {a}"
        );
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
        let (params, update, starts) = fixture_params();
        let after = mem::block_mass_after_update(&params, &update, &starts, 1);
        assert_eq!(
            mem::l1_mass(&params[..2]),
            mem::l1_mass(&after[..2]),
            "block 0 mass must be preserved"
        );
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // If we update the WHOLE param vec instead of just the block,
        // inactive-block mass is no longer preserved (use a non-symmetric
        // update so |p+u| ≠ |p|).
        let (params, _, _) = fixture_params();
        let asym_update = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let bogus_after: Vec<f64> = params
            .iter()
            .zip(asym_update.iter())
            .map(|(p, u)| p + u)
            .collect();
        assert_ne!(mem::l1_mass(&params[..2]), mem::l1_mass(&bogus_after[..2]));
    }

    #[test]
    fn deterministic_across_runs() {
        let (params, update, starts) = fixture_params();
        let a = mem::block_mass_after_update(&params, &update, &starts, 1);
        let b = mem::block_mass_after_update(&params, &update, &starts, 1);
        assert_eq!(a, b);
    }
}
