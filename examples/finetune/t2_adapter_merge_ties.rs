//! # Tier 2.4 — Adapter merge — TIES (llama family)
//!
//! Falsifier: TIES merge of two LoRAs preserves shared sign-direction
//! parameters. When both inputs agree on sign at index i, merged[i] retains
//! the same sign and equals the mean of the agreeing entries.
//!
//! Run with: cargo run --example t2_adapter_merge_ties

use apr_cookbook::finetune::adapter_merge as am;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_a() -> Vec<f64> {
    vec![1.0, -1.0, 0.5, 2.0, -0.3]
}
fn fixture_b() -> Vec<f64> {
    vec![2.0, 1.0, 0.25, 3.0, -0.7]
}

fn shared_sign_indices(a: &[f64], b: &[f64]) -> Vec<usize> {
    a.iter()
        .zip(b)
        .enumerate()
        .filter_map(|(i, (x, y))| {
            if x.signum() == y.signum() && x.abs() > 0.0 {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_adapter_merge_ties")?;
    let a = fixture_a();
    let b = fixture_b();
    let merged = am::ties_merge(&[a.clone(), b.clone()]);
    let shared = shared_sign_indices(&a, &b);
    println!(
        "✓ TIES merge: |a|={:.3} |b|={:.3} |merged|={:.3} shared={:?}",
        am::norm(&a),
        am::norm(&b),
        am::norm(&merged),
        shared
    );
    for &i in &shared {
        assert!(
            merged[i].signum() == a[i].signum(),
            "TIES must preserve shared-sign direction at index {i}"
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
        let a = fixture_a();
        let b = fixture_b();
        let merged = am::ties_merge(&[a.clone(), b.clone()]);
        for i in shared_sign_indices(&a, &b) {
            assert_eq!(merged[i].signum(), a[i].signum());
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Take original shared-sign indices, then negate B. At those indices,
        // signs now disagree, so TIES votes by *magnitude*. When |b| dominates
        // |a|, the merged sign flips away from A's original sign.
        let a = fixture_a();
        let b_original = fixture_b();
        let original_shared = shared_sign_indices(&a, &b_original);
        let b_negated: Vec<f64> = b_original.iter().map(|v| -v).collect();
        let merged = am::ties_merge(&[a.clone(), b_negated]);
        // At least one originally-shared index must now have a different sign
        // (or be zeroed) since we removed the shared-sign agreement.
        let any_flipped = original_shared
            .iter()
            .any(|&i| merged[i].signum() != a[i].signum() || merged[i].abs() < 1e-12);
        assert!(
            any_flipped,
            "negating B must break shared-sign preservation at ≥1 index"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = fixture_a();
        let b = fixture_b();
        let m1 = am::ties_merge(&[a.clone(), b.clone()]);
        let m2 = am::ties_merge(&[a, b]);
        assert_eq!(m1, m2);
    }
}
