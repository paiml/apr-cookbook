//! # apr monitor --quantiles — Sliding-Window Quantile Picker
//!
//! `apr monitor --quantiles <Q,Q,...>` reports per-window quantiles
//! (default p50,p95,p99). Constraints: each q ∈ (0, 1); duplicates
//! collapsed; max 10 quantiles to keep TUI rendering legible. This
//! recipe builds the validator + dedup.
//!
//! Demonstrates the **MON.4** recipe for PMAT-114 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MON-001 + Cormode et al. 2010 (sliding quantiles)
//!
//! Run with: cargo run --example cli_monitor_quantile_picker
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_QUANTILES: usize = 10;

#[derive(Debug, PartialEq)]
pub enum QuantileVerdict {
    Ok(Vec<f64>),
    OutOfRange { value: f64 },
    TooMany { count: usize },
    Empty,
}

pub fn parse_and_validate(input: &[f64]) -> QuantileVerdict {
    if input.is_empty() {
        return QuantileVerdict::Empty;
    }
    for &q in input {
        if !q.is_finite() || q <= 0.0 || q >= 1.0 {
            return QuantileVerdict::OutOfRange { value: q };
        }
    }
    let mut sorted: Vec<f64> = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if sorted.len() > MAX_QUANTILES {
        return QuantileVerdict::TooMany {
            count: sorted.len(),
        };
    }
    QuantileVerdict::Ok(sorted)
}

pub fn defaults() -> Vec<f64> {
    vec![0.50, 0.95, 0.99]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_quantile_picker")?;

    let cases = [
        ("defaults", defaults()),
        ("with dup", vec![0.5, 0.5, 0.95]),
        ("bad", vec![0.5, 1.5]),
        (
            "too many",
            (0..15).map(|i| 0.05 + 0.05 * i as f64).collect(),
        ),
    ];
    for (label, input) in cases {
        println!("{label:>12}  →  {:?}", parse_and_validate(&input));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn defaults_pass() {
        assert!(matches!(
            parse_and_validate(&defaults()),
            QuantileVerdict::Ok(_)
        ));
    }

    #[test]
    fn duplicates_collapsed() {
        let v = parse_and_validate(&[0.5, 0.5, 0.95]);
        if let QuantileVerdict::Ok(qs) = v {
            assert_eq!(qs.len(), 2);
        }
    }

    #[test]
    fn out_of_range_rejected() {
        assert!(matches!(
            parse_and_validate(&[0.5, 1.5]),
            QuantileVerdict::OutOfRange { .. }
        ));
        assert!(matches!(
            parse_and_validate(&[0.5, -0.1]),
            QuantileVerdict::OutOfRange { .. }
        ));
        // q=0 and q=1 are exclusive.
        assert!(matches!(
            parse_and_validate(&[0.0]),
            QuantileVerdict::OutOfRange { .. }
        ));
        assert!(matches!(
            parse_and_validate(&[1.0]),
            QuantileVerdict::OutOfRange { .. }
        ));
    }

    #[test]
    fn nan_rejected() {
        assert!(matches!(
            parse_and_validate(&[f64::NAN]),
            QuantileVerdict::OutOfRange { .. }
        ));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(parse_and_validate(&[]), QuantileVerdict::Empty);
    }

    #[test]
    fn too_many_rejected() {
        let many: Vec<f64> = (1..=15).map(|i| f64::from(i) * 0.05).collect();
        assert!(matches!(
            parse_and_validate(&many),
            QuantileVerdict::TooMany { .. }
        ));
    }

    #[test]
    fn output_is_sorted() {
        let v = parse_and_validate(&[0.99, 0.5, 0.95]);
        if let QuantileVerdict::Ok(qs) = v {
            for w in qs.windows(2) {
                assert!(w[0] < w[1], "not sorted: {qs:?}");
            }
        }
    }

    #[test]
    fn at_max_quantiles_passes() {
        let exactly_max: Vec<f64> = (1..=MAX_QUANTILES)
            .map(|i| f64::from(i as u32) * 0.09)
            .collect();
        assert!(matches!(
            parse_and_validate(&exactly_max),
            QuantileVerdict::Ok(_)
        ));
    }
}
