//! # apr merge --strategy ties --density — Density Gate
//!
//! `apr merge --strategy ties --density <D>` keeps the top D fraction
//! of magnitude-sorted parameters per task vector. D must be in (0, 1];
//! D=1 falls back to weighted-average; D < 0.05 typically destroys the
//! merged model. This recipe builds the density envelope.
//!
//! Demonstrates the **MERGE.4** recipe for PMAT-112 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-001 + Yadav et al. 2023 (TIES-Merging)
//!
//! Run with: cargo run --example cli_merge_ties_density_gate
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DensityVerdict {
    Ok,
    DegeneratesToAverage, // D == 1.0
    BelowSafetyFloor,     // D < 0.05
    OutOfRange,           // D <= 0 or > 1
}

const SAFETY_FLOOR: f64 = 0.05;

pub fn classify(density: f64) -> DensityVerdict {
    if !density.is_finite() || density <= 0.0 || density > 1.0 {
        return DensityVerdict::OutOfRange;
    }
    if (density - 1.0).abs() < f64::EPSILON {
        return DensityVerdict::DegeneratesToAverage;
    }
    if density < SAFETY_FLOOR {
        return DensityVerdict::BelowSafetyFloor;
    }
    DensityVerdict::Ok
}

pub fn params_kept(num_params: u64, density: f64) -> Option<u64> {
    if !density.is_finite() || density <= 0.0 || density > 1.0 {
        return None;
    }
    Some((num_params as f64 * density).round() as u64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_ties_density_gate")?;

    for d in [0.0, 0.04, 0.20, 0.50, 1.0, 1.2, f64::NAN] {
        let v = classify(d);
        let kept = params_kept(7_000_000_000, d);
        println!("D={d:>6.2}  →  {v:?}   kept(7B) = {kept:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn density_one_degenerates_to_average() {
        // D=1 keeps everything → equivalent to plain weighted average.
        assert_eq!(classify(1.0), DensityVerdict::DegeneratesToAverage);
    }

    #[test]
    fn typical_density_ok() {
        assert_eq!(classify(0.20), DensityVerdict::Ok);
        assert_eq!(classify(0.50), DensityVerdict::Ok);
    }

    #[test]
    fn below_safety_floor_rejected() {
        assert_eq!(classify(0.01), DensityVerdict::BelowSafetyFloor);
        assert_eq!(classify(0.049), DensityVerdict::BelowSafetyFloor);
    }

    #[test]
    fn at_safety_floor_passes() {
        // 0.05 is the minimum acceptable.
        assert_eq!(classify(SAFETY_FLOOR), DensityVerdict::Ok);
    }

    #[test]
    fn zero_or_negative_out_of_range() {
        assert_eq!(classify(0.0), DensityVerdict::OutOfRange);
        assert_eq!(classify(-0.5), DensityVerdict::OutOfRange);
    }

    #[test]
    fn over_one_out_of_range() {
        assert_eq!(classify(1.01), DensityVerdict::OutOfRange);
        assert_eq!(classify(2.0), DensityVerdict::OutOfRange);
    }

    #[test]
    fn nan_or_inf_out_of_range() {
        assert_eq!(classify(f64::NAN), DensityVerdict::OutOfRange);
        assert_eq!(classify(f64::INFINITY), DensityVerdict::OutOfRange);
    }

    #[test]
    fn params_kept_proportional() {
        // 1B params at D=0.2 → 200M kept.
        assert_eq!(params_kept(1_000_000_000, 0.2), Some(200_000_000));
    }
}
