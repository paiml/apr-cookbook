//! # apr merge --strategy slerp --t — Interpolation Parameter Clamp
//!
//! `apr merge --strategy slerp --t <T>` interpolates between two models
//! along the great-circle arc in weight space. T=0 returns model A, T=1
//! returns model B, T=0.5 is the midpoint. Out-of-range T silently
//! extrapolates beyond the arc — usually catastrophic. This recipe
//! enforces the [0, 1] envelope and clamps with a warning.
//!
//! Demonstrates the **MERGE.5** recipe for PMAT-112 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-001 + Shoemake 1985 (SLERP)
//!
//! Run with: cargo run --example cli_merge_slerp_t_clamp
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ClampVerdict {
    Ok,
    ClampedFromAbove { original: f64 },
    ClampedFromBelow { original: f64 },
    Rejected, // NaN/inf
}

#[derive(Debug, PartialEq)]
pub struct ClampResult {
    pub t: f64,
    pub verdict: ClampVerdict,
}

pub fn clamp_t(t: f64) -> ClampResult {
    if !t.is_finite() {
        return ClampResult {
            t: 0.5,
            verdict: ClampVerdict::Rejected,
        };
    }
    if t < 0.0 {
        return ClampResult {
            t: 0.0,
            verdict: ClampVerdict::ClampedFromBelow { original: t },
        };
    }
    if t > 1.0 {
        return ClampResult {
            t: 1.0,
            verdict: ClampVerdict::ClampedFromAbove { original: t },
        };
    }
    ClampResult {
        t,
        verdict: ClampVerdict::Ok,
    }
}

pub fn dominant_model(t: f64) -> &'static str {
    let r = clamp_t(t);
    if r.t < 0.5 {
        "A"
    } else if r.t > 0.5 {
        "B"
    } else {
        "midpoint"
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_slerp_t_clamp")?;

    for t in [0.0, 0.25, 0.5, 0.75, 1.0, -0.1, 1.5, f64::NAN] {
        println!(
            "t={t:>6.2}  →  {:?}   dominant={}",
            clamp_t(t),
            dominant_model(t)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn in_range_passes_unchanged() {
        let r = clamp_t(0.3);
        assert_eq!(r.t, 0.3);
        assert_eq!(r.verdict, ClampVerdict::Ok);
    }

    #[test]
    fn negative_clamped_to_zero() {
        let r = clamp_t(-0.1);
        assert_eq!(r.t, 0.0);
        assert!(matches!(r.verdict, ClampVerdict::ClampedFromBelow { .. }));
    }

    #[test]
    fn over_one_clamped_to_one() {
        let r = clamp_t(1.5);
        assert_eq!(r.t, 1.0);
        assert!(matches!(r.verdict, ClampVerdict::ClampedFromAbove { .. }));
    }

    #[test]
    fn nan_rejected_to_midpoint() {
        let r = clamp_t(f64::NAN);
        assert_eq!(r.t, 0.5);
        assert_eq!(r.verdict, ClampVerdict::Rejected);
    }

    #[test]
    fn infinity_rejected() {
        assert_eq!(clamp_t(f64::INFINITY).verdict, ClampVerdict::Rejected);
        assert_eq!(clamp_t(f64::NEG_INFINITY).verdict, ClampVerdict::Rejected);
    }

    #[test]
    fn boundary_values_pass() {
        assert_eq!(clamp_t(0.0).verdict, ClampVerdict::Ok);
        assert_eq!(clamp_t(1.0).verdict, ClampVerdict::Ok);
    }

    #[test]
    fn dominant_at_boundaries() {
        assert_eq!(dominant_model(0.0), "A");
        assert_eq!(dominant_model(1.0), "B");
        assert_eq!(dominant_model(0.5), "midpoint");
    }

    #[test]
    fn dominant_after_clamp_consistent() {
        // -0.5 clamps to 0.0 → dominant A.
        // 1.5 clamps to 1.0 → dominant B.
        assert_eq!(dominant_model(-0.5), "A");
        assert_eq!(dominant_model(1.5), "B");
    }
}
