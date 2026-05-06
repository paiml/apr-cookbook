//! # apr diagnose --grad-scan — NaN/Inf Gradient Scanner
//!
//! Training instability often shows as NaN or ±inf gradients. `apr
//! diagnose --grad-scan` walks per-parameter gradients and reports
//! the first offending parameter (genchi-genbutsu: stop at first
//! defect). Returns `Healthy` if all finite. This recipe builds the
//! scanner.
//!
//! Demonstrates the **DIAG.4** recipe for PMAT-116 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAG-001 + Toyota Way (genchi genbutsu)
//!
//! Run with: cargo run --example cli_diagnose_grad_nan_scanner
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GradVerdict {
    Healthy,
    HasNan {
        param: String,
        index: usize,
    },
    HasInf {
        param: String,
        index: usize,
        sign: i8,
    },
    Empty,
}

pub fn scan(name: &str, grads: &[f64]) -> GradVerdict {
    if grads.is_empty() {
        return GradVerdict::Empty;
    }
    for (i, g) in grads.iter().enumerate() {
        if g.is_nan() {
            return GradVerdict::HasNan {
                param: name.into(),
                index: i,
            };
        }
        if g.is_infinite() {
            return GradVerdict::HasInf {
                param: name.into(),
                index: i,
                sign: if *g > 0.0 { 1 } else { -1 },
            };
        }
    }
    GradVerdict::Healthy
}

pub fn scan_all(named: &[(&str, &[f64])]) -> GradVerdict {
    for (name, grads) in named {
        let v = scan(name, grads);
        if !matches!(v, GradVerdict::Healthy | GradVerdict::Empty) {
            return v;
        }
    }
    GradVerdict::Healthy
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_grad_nan_scanner")?;

    let healthy: &[f64] = &[0.01, -0.02, 0.03];
    let with_nan: &[f64] = &[0.01, f64::NAN, 0.03];
    let with_inf: &[f64] = &[0.01, f64::NEG_INFINITY, 0.03];
    println!("healthy: {:?}", scan("layer.0.weight", healthy));
    println!("nan:     {:?}", scan("layer.5.weight", with_nan));
    println!("inf:     {:?}", scan("layer.7.bias", with_inf));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_finite_healthy() {
        assert_eq!(scan("p", &[0.01, -0.02, 0.03]), GradVerdict::Healthy);
    }

    #[test]
    fn nan_detected_with_index() {
        let v = scan("p", &[0.0, 0.0, f64::NAN, 0.0]);
        assert!(matches!(v, GradVerdict::HasNan { index: 2, .. }));
    }

    #[test]
    fn pos_inf_detected_with_sign() {
        let v = scan("p", &[f64::INFINITY]);
        assert!(matches!(v, GradVerdict::HasInf { sign: 1, .. }));
    }

    #[test]
    fn neg_inf_detected_with_sign() {
        let v = scan("p", &[f64::NEG_INFINITY]);
        assert!(matches!(v, GradVerdict::HasInf { sign: -1, .. }));
    }

    #[test]
    fn empty_returns_empty() {
        assert_eq!(scan("p", &[]), GradVerdict::Empty);
    }

    #[test]
    fn first_offender_returned() {
        // NaN at idx 1, Inf at idx 3 — should report NaN first.
        let v = scan("p", &[0.0, f64::NAN, 0.0, f64::INFINITY]);
        assert!(matches!(v, GradVerdict::HasNan { index: 1, .. }));
    }

    #[test]
    fn scan_all_short_circuits_on_first_defect() {
        let healthy: &[f64] = &[0.0, 0.1];
        let bad: &[f64] = &[0.0, f64::NAN];
        let later: &[f64] = &[0.0, 0.2];
        let v = scan_all(&[("layer.0", healthy), ("layer.5", bad), ("layer.10", later)]);
        if let GradVerdict::HasNan { param, .. } = v {
            assert_eq!(param, "layer.5");
        } else {
            panic!("expected HasNan at layer.5");
        }
    }

    #[test]
    fn scan_all_healthy_when_all_clean() {
        let a: &[f64] = &[0.0, 0.1];
        let b: &[f64] = &[0.5, -0.3];
        assert_eq!(scan_all(&[("p1", a), ("p2", b)]), GradVerdict::Healthy);
    }
}
