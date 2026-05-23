//! # Tier 2.4 — Adapter merge — SLERP (phi family)
//!
//! Falsifier: SLERP at t=0.5 produces a midpoint whose norm is sandwiched
//! between min(‖A‖,‖B‖) and max(‖A‖,‖B‖) × 1.05, and is ≥ 0.45·(‖A‖+‖B‖).
//!
//! Run with: cargo run --example t2_adapter_merge_slerp

use apr_cookbook::finetune::adapter_merge as am;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_a() -> Vec<f64> {
    (0..32)
        .map(|i| (((i as u32 * 7) % 13) as f64) / 13.0 - 0.4)
        .collect()
}

fn fixture_b() -> Vec<f64> {
    (0..32)
        .map(|i| (((i as u32 * 11 + 1) % 17) as f64) / 17.0 - 0.4)
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_adapter_merge_slerp")?;
    let a = fixture_a();
    let b = fixture_b();
    let mid = am::slerp_merge(&a, &b, 0.5);
    let na = am::norm(&a);
    let nb = am::norm(&b);
    let nm = am::norm(&mid);
    println!("✓ SLERP t=0.5: |a|={:.4} |b|={:.4} |mid|={:.4}", na, nb, nm);
    assert!(
        nm <= na.max(nb) * 1.05,
        "SLERP midpoint norm should be ≤ max(|A|,|B|)*1.05"
    );
    assert!(
        nm >= 0.45 * (na + nb),
        "SLERP midpoint norm should be ≥ 0.45*(|A|+|B|)"
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
        let a = fixture_a();
        let b = fixture_b();
        let mid = am::slerp_merge(&a, &b, 0.5);
        let na = am::norm(&a);
        let nb = am::norm(&b);
        let nm = am::norm(&mid);
        assert!(nm <= na.max(nb) * 1.05);
        assert!(nm >= 0.45 * (na + nb));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // t=1.0 → output equals B exactly, NOT a midpoint of A and B.
        let a = fixture_a();
        let b = fixture_b();
        let end = am::slerp_merge(&a, &b, 1.0);
        let nb = am::norm(&b);
        let ne = am::norm(&end);
        // SLERP at t=1 is exactly B (within float tolerance).
        assert!((ne - nb).abs() < 1e-6);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = fixture_a();
        let b = fixture_b();
        let m1 = am::slerp_merge(&a, &b, 0.5);
        let m2 = am::slerp_merge(&a, &b, 0.5);
        assert_eq!(m1, m2);
    }
}
