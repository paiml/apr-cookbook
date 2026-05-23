//! # Tier 2.5 — Orthogonal Fine-Tuning (OFT) — phi family
//!
//! Falsifier: trained R matrix satisfies R^T R = I within ε=1e-4 across
//! training. Implemented via Givens-style block-rotation parameterization
//! that is exactly orthogonal by construction for any θ.
//!
//! Run with: cargo run --example t2_oft

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const D: usize = 8;
const EPS: f64 = 1e-4;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_oft")?;
    // Sweep θ ∈ {0.1, 0.4, 1.0, 1.57}; orthogonality must hold throughout.
    for theta in &[0.1_f64, 0.4, 1.0, std::f64::consts::FRAC_PI_2] {
        let r = peft::oft_orthogonal_rotation(D, *theta);
        assert!(
            peft::is_orthogonal(&r, EPS),
            "OFT rotation at θ={theta} must satisfy R^T R = I within ε={EPS}"
        );
    }
    println!("✓ OFT: orthogonality preserved across θ ∈ [0.1, π/2] (ε={EPS})");
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
        for theta in &[0.1_f64, 0.4, 1.0, std::f64::consts::FRAC_PI_2] {
            let r = peft::oft_orthogonal_rotation(D, *theta);
            assert!(peft::is_orthogonal(&r, EPS));
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let mut r = peft::oft_orthogonal_rotation(D, 0.4);
        r[0][0] += 0.5;
        assert!(!peft::is_orthogonal(&r, EPS));
    }

    #[test]
    fn deterministic_across_runs() {
        let r1 = peft::oft_orthogonal_rotation(D, 0.4);
        let r2 = peft::oft_orthogonal_rotation(D, 0.4);
        assert_eq!(r1, r2);
    }
}
