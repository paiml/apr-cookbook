//! # apr merge --strategy weighted — `--weights <CSV>` Parser
//!
//! `apr merge --strategy weighted --weights "0.7,0.3"` accepts a CSV of
//! per-model weights. This recipe builds the parser and asserts the
//! contract: weights must be finite, non-negative, count == n_models,
//! sum normalises to 1.0 (within FP slack); a leading 0 weight is allowed
//! (effectively excludes the model from the merge).
//!
//! Demonstrates the **MERGE.9** recipe for PMAT-105 (apr merge coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MERGE-002
//!
//! Run with: cargo run --example cli_merge_weights_csv_parser
//!
//! Added by PMAT-105 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WeightsVerdict {
    Ok { normalized: Vec<f64> },
    CountMismatch { expected: usize, observed: usize },
    NonFiniteWeight { index: usize },
    NegativeWeight { index: usize, value: f64 },
    AllZero,
    MalformedToken { index: usize, raw: String },
}

pub fn parse_weights(raw: &str, n_models: usize) -> WeightsVerdict {
    let tokens: Vec<&str> = raw.split(',').map(str::trim).collect();
    if tokens.len() != n_models {
        return WeightsVerdict::CountMismatch {
            expected: n_models,
            observed: tokens.len(),
        };
    }
    let mut parsed: Vec<f64> = Vec::with_capacity(n_models);
    for (i, t) in tokens.iter().enumerate() {
        let Ok(v) = t.parse::<f64>() else {
            return WeightsVerdict::MalformedToken {
                index: i,
                raw: (*t).to_string(),
            };
        };
        if !v.is_finite() {
            return WeightsVerdict::NonFiniteWeight { index: i };
        }
        if v < 0.0 {
            return WeightsVerdict::NegativeWeight { index: i, value: v };
        }
        parsed.push(v);
    }
    let sum: f64 = parsed.iter().sum();
    if sum <= 0.0 {
        return WeightsVerdict::AllZero;
    }
    let normalized: Vec<f64> = parsed.iter().map(|w| w / sum).collect();
    WeightsVerdict::Ok { normalized }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_merge_weights_csv_parser")?;

    let cases = [
        ("0.7,0.3", 2),
        ("1,1,1", 3),
        ("0.5", 2), // count mismatch
        ("0.5,inf", 2),
        ("0.5,-0.1", 2),
        ("0,0,0", 3),
        ("1.0,abc", 2),
    ];

    for (raw, n) in cases {
        println!("{raw:>15} ({n} models)  →  {:?}", parse_weights(raw, n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_normalises_to_unit_sum() {
        let v = parse_weights("0.7,0.3", 2);
        if let WeightsVerdict::Ok { normalized } = v {
            assert!((normalized[0] - 0.7).abs() < 1e-9);
            assert!((normalized[1] - 0.3).abs() < 1e-9);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn unnormalized_input_normalizes_to_unit_sum() {
        let v = parse_weights("2,3", 2);
        if let WeightsVerdict::Ok { normalized } = v {
            assert!((normalized[0] - 0.4).abs() < 1e-9);
            assert!((normalized[1] - 0.6).abs() < 1e-9);
        }
    }

    #[test]
    fn count_mismatch_rejected() {
        assert_eq!(
            parse_weights("0.5", 2),
            WeightsVerdict::CountMismatch {
                expected: 2,
                observed: 1,
            }
        );
    }

    #[test]
    fn negative_weight_rejected() {
        let v = parse_weights("0.5,-0.1", 2);
        assert!(matches!(v, WeightsVerdict::NegativeWeight { index: 1, .. }));
    }

    #[test]
    fn nonfinite_weight_rejected() {
        let v = parse_weights("0.5,inf", 2);
        assert!(matches!(v, WeightsVerdict::NonFiniteWeight { index: 1 }));
    }

    #[test]
    fn all_zero_rejected() {
        // Avoids divide-by-zero in normalization.
        let v = parse_weights("0,0,0", 3);
        assert_eq!(v, WeightsVerdict::AllZero);
    }

    #[test]
    fn malformed_token_returns_malformed() {
        let v = parse_weights("1.0,abc", 2);
        assert!(matches!(v, WeightsVerdict::MalformedToken { index: 1, .. }));
    }

    #[test]
    fn zero_weight_for_one_model_normalizes_others() {
        // 0 weight excludes that model — others normalize among themselves.
        let v = parse_weights("0,1,1", 3);
        if let WeightsVerdict::Ok { normalized } = v {
            assert!((normalized[0] - 0.0).abs() < 1e-9);
            assert!((normalized[1] - 0.5).abs() < 1e-9);
            assert!((normalized[2] - 0.5).abs() < 1e-9);
        }
    }
}
