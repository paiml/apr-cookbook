//! # Visualization Axis Scale Classifier
//!
//! Pick the right axis scale by data range: Linear for span < 100×;
//! Log when max/min > 100× (positive only); SymLog for data
//! straddling zero with wide range (e.g., signed gradients). This
//! recipe builds the classifier.
//!
//! Demonstrates the **VIZ.3** recipe for PMAT-128 (visualization coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cleveland (1985). The Elements of Graphing Data §3.
//!
//! Run with: cargo run --example viz_axis_scale_classifier
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisScale {
    Linear,
    Log,
    SymLog,
}

#[derive(Debug, PartialEq)]
pub enum ScaleVerdict {
    Ok(AxisScale),
    EmptyData,
    InvalidValues,
}

const WIDE_RANGE_RATIO: f64 = 100.0;

pub fn classify(values: &[f64]) -> ScaleVerdict {
    if values.is_empty() {
        return ScaleVerdict::EmptyData;
    }
    if values.iter().any(|x| !x.is_finite()) {
        return ScaleVerdict::InvalidValues;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let crosses_zero = min < 0.0 && max > 0.0;
    let abs_max = max.abs().max(min.abs());
    let abs_min_nonzero = values
        .iter()
        .copied()
        .filter(|x| *x != 0.0)
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min);
    if !abs_min_nonzero.is_finite() {
        return ScaleVerdict::Ok(AxisScale::Linear);
    }
    let span_ratio = abs_max / abs_min_nonzero;
    if crosses_zero && span_ratio > WIDE_RANGE_RATIO {
        return ScaleVerdict::Ok(AxisScale::SymLog);
    }
    if !crosses_zero && min > 0.0 && span_ratio > WIDE_RANGE_RATIO {
        return ScaleVerdict::Ok(AxisScale::Log);
    }
    ScaleVerdict::Ok(AxisScale::Linear)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("viz_axis_scale_classifier")?;

    let cases: &[(&str, &[f64])] = &[
        ("linear narrow", &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ("log positive wide", &[0.001, 0.01, 0.1, 1.0, 100.0]),
        ("symlog signed wide", &[-100.0, -1.0, 0.0, 1.0, 100.0]),
        ("nan", &[1.0, f64::NAN]),
        ("empty", &[]),
    ];
    for (label, vs) in cases {
        println!("{label:<22}  →  {:?}", classify(vs));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn narrow_linear_data() {
        let v = classify(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Linear));
    }

    #[test]
    fn wide_positive_log() {
        let v = classify(&[0.001, 0.01, 1.0, 100.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Log));
    }

    #[test]
    fn signed_wide_symlog() {
        // Span ratio strictly > 100; need 0.1..1000 to trigger SymLog.
        let v = classify(&[-1000.0, -0.1, 0.0, 0.1, 1000.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::SymLog));
    }

    #[test]
    fn signed_narrow_linear() {
        // Crosses zero but ratio not wide.
        let v = classify(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Linear));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(classify(&[]), ScaleVerdict::EmptyData);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(classify(&[1.0, f64::NAN]), ScaleVerdict::InvalidValues);
    }

    #[test]
    fn infinity_rejected() {
        assert_eq!(classify(&[1.0, f64::INFINITY]), ScaleVerdict::InvalidValues);
    }

    #[test]
    fn all_zero_linear() {
        // No nonzero values → linear.
        let v = classify(&[0.0, 0.0, 0.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Linear));
    }

    #[test]
    fn boundary_at_100x_linear() {
        // 1.0 to 100.0 = 100× ratio (not strictly > 100 → Linear).
        let v = classify(&[1.0, 100.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Linear));
    }

    #[test]
    fn just_over_100x_log() {
        let v = classify(&[1.0, 101.0]);
        assert_eq!(v, ScaleVerdict::Ok(AxisScale::Log));
    }
}
