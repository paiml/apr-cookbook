//! # apr merge --strategy dare — `--drop-rate` Band Validator
//!
//! `apr merge --strategy dare --drop-rate <P>` accepts P ∈ [0.5, 0.99]
//! (Yu et al. 2024 default 0.9). Below 0.5 makes DARE behave like a
//! trivial weighted merge; ≥ 1.0 drops every weight (broken model). This
//! recipe builds the validator with seed reproducibility check.
//!
//! Demonstrates the **MERGE.10** recipe for PMAT-105 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-003 + Yu et al. (2024) DARE
//!
//! Run with: cargo run --example cli_merge_dare_drop_rate_band
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DareVerdict {
    Ok { effective_density: f64 },
    BelowFloor { observed: f64, floor: f64 },
    AboveCeiling { observed: f64, ceiling: f64 },
    NotFinite,
}

const FLOOR: f64 = 0.5;
const CEILING: f64 = 0.99;

pub fn validate_drop_rate(p: f64) -> DareVerdict {
    if !p.is_finite() {
        return DareVerdict::NotFinite;
    }
    if p < FLOOR {
        return DareVerdict::BelowFloor {
            observed: p,
            floor: FLOOR,
        };
    }
    if p > CEILING {
        return DareVerdict::AboveCeiling {
            observed: p,
            ceiling: CEILING,
        };
    }
    DareVerdict::Ok {
        effective_density: 1.0 - p,
    }
}

pub fn seed_is_reproducible(seed: u32) -> bool {
    // Trivial check: any non-zero seed is reproducible.
    seed != 0
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_dare_drop_rate_band")?;

    for p in [0.0_f64, 0.3, 0.5, 0.9, 0.99, 1.0, 1.5, f64::NAN] {
        println!("--drop-rate {p:>5.2}  →  {:?}", validate_drop_rate(p));
    }
    println!("\nseed=0  reproducible? {}", seed_is_reproducible(0));
    println!("seed=42 reproducible? {}", seed_is_reproducible(42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn default_0_9_passes() {
        let v = validate_drop_rate(0.9);
        if let DareVerdict::Ok { effective_density } = v {
            assert!((effective_density - 0.1).abs() < 1e-9);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn below_0_5_rejected() {
        // Too low — DARE collapses to trivial weighted merge.
        assert!(matches!(
            validate_drop_rate(0.3),
            DareVerdict::BelowFloor { .. }
        ));
        assert!(matches!(
            validate_drop_rate(0.0),
            DareVerdict::BelowFloor { .. }
        ));
    }

    #[test]
    fn boundary_at_0_5_passes() {
        // Conservative-pass at the floor.
        assert!(matches!(validate_drop_rate(0.5), DareVerdict::Ok { .. }));
    }

    #[test]
    fn above_0_99_rejected() {
        assert!(matches!(
            validate_drop_rate(1.0),
            DareVerdict::AboveCeiling { .. }
        ));
        assert!(matches!(
            validate_drop_rate(1.5),
            DareVerdict::AboveCeiling { .. }
        ));
    }

    #[test]
    fn boundary_at_0_99_passes() {
        assert!(matches!(validate_drop_rate(0.99), DareVerdict::Ok { .. }));
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(validate_drop_rate(f64::NAN), DareVerdict::NotFinite);
    }

    #[test]
    fn inf_rejected() {
        assert_eq!(validate_drop_rate(f64::INFINITY), DareVerdict::NotFinite);
    }

    #[test]
    fn seed_zero_rejected_as_nonreproducible() {
        // Seed 0 is the "default unset" sentinel and would behave non-deterministically
        // in some PRNGs. Refuse rather than let the operator get surprises.
        assert!(!seed_is_reproducible(0));
    }

    #[test]
    fn nonzero_seed_passes() {
        assert!(seed_is_reproducible(42));
        assert!(seed_is_reproducible(1));
    }
}
