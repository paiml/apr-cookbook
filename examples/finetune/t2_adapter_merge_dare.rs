//! # Tier 2.4 — Adapter merge — DARE (mistral family)
//!
//! Falsifier: DARE merge with drop_p=0.5 reduces the parameter count by
//! ≈ 50% (deterministic-stride mask, ±5% tolerance) and rescales the
//! survivors so the *expected* magnitude is preserved.
//!
//! Run with: cargo run --example t2_adapter_merge_dare

use apr_cookbook::finetune::adapter_merge as am;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DROP_P: f64 = 0.5;
const N: usize = 200;

fn fixture() -> Vec<f64> {
    (0..N)
        .map(|i| (((i as u32 * 7 + 3) % 19) as f64) / 19.0 - 0.4)
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_adapter_merge_dare")?;
    let delta = fixture();
    let merged = am::dare_merge(&delta, DROP_P);
    let zeros = merged.iter().filter(|v| v.abs() < 1e-12).count();
    let target = (N as f64 * DROP_P) as usize;
    let tol = (N as f64 * 0.05) as usize;
    println!(
        "✓ DARE merge p={}: {}/{} zeros (target={}±{})",
        DROP_P, zeros, N, target, tol
    );
    assert!(
        zeros >= target - tol && zeros <= target + tol,
        "DARE drop_p=0.5 must zero ~50% of entries (got {zeros}/{N})"
    );
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
        let delta = fixture();
        let merged = am::dare_merge(&delta, DROP_P);
        let zeros = merged.iter().filter(|v| v.abs() < 1e-12).count();
        let target = (N as f64 * DROP_P) as usize;
        let tol = (N as f64 * 0.05) as usize;
        assert!((target - tol..=target + tol).contains(&zeros));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // drop_p=0.0 means no entries should be zeroed — falsifier (≥45% zeros)
        // must NOT hold.
        let delta = fixture();
        let merged = am::dare_merge(&delta, 0.0);
        let zeros = merged.iter().filter(|v| v.abs() < 1e-12).count();
        assert!(zeros < N / 4, "drop_p=0 must not zero out half the entries");
    }

    #[test]
    fn deterministic_across_runs() {
        let delta = fixture();
        let m1 = am::dare_merge(&delta, DROP_P);
        let m2 = am::dare_merge(&delta, DROP_P);
        assert_eq!(m1, m2);
    }
}
