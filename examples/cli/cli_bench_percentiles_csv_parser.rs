//! # apr bench — `--percentiles` CSV Parser (CRUX-E-07)
//!
//! `apr bench --percentiles 50,95,99` controls which latency percentiles
//! the JSON report includes. This recipe builds the CSV parser and
//! enforces the contract: values in (0, 100], deduped, sorted ascending.
//!
//! Demonstrates the **BENCH.10** recipe for PMAT-109 (apr bench coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-E-07
//!
//! Run with: cargo run --example cli_bench_percentiles_csv_parser
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum PercentileVerdict {
    Ok(Vec<f64>),
    OutOfBand { value: f64 },
    MalformedToken { raw: String },
    Empty,
}

pub fn parse_percentiles(s: &str) -> PercentileVerdict {
    let tokens: Vec<&str> = s
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return PercentileVerdict::Empty;
    }
    let mut set: BTreeSet<u64> = BTreeSet::new(); // store as fixed-point (×100)
    for raw in &tokens {
        let Ok(v) = raw.parse::<f64>() else {
            return PercentileVerdict::MalformedToken {
                raw: (*raw).to_string(),
            };
        };
        if !v.is_finite() || v <= 0.0 || v > 100.0 {
            return PercentileVerdict::OutOfBand { value: v };
        }
        set.insert((v * 100.0) as u64);
    }
    PercentileVerdict::Ok(set.iter().map(|n| *n as f64 / 100.0).collect())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_bench_percentiles_csv_parser")?;

    for s in [
        "50,95,99",
        "99,50,95",
        "50.5,99.9",
        "50,100",
        "50,150",
        "abc",
        "",
    ] {
        println!("--percentiles {s:>15}  →  {:?}", parse_percentiles(s));
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
    fn default_50_95_99_passes() {
        let v = parse_percentiles("50,95,99");
        assert_eq!(v, PercentileVerdict::Ok(vec![50.0, 95.0, 99.0]));
    }

    #[test]
    fn out_of_order_input_sorts_ascending() {
        let v = parse_percentiles("99,50,95");
        assert_eq!(v, PercentileVerdict::Ok(vec![50.0, 95.0, 99.0]));
    }

    #[test]
    fn duplicates_deduped() {
        let v = parse_percentiles("50,50,50");
        assert_eq!(v, PercentileVerdict::Ok(vec![50.0]));
    }

    #[test]
    fn fractional_percentiles_supported() {
        let v = parse_percentiles("50.5,99.9");
        assert_eq!(v, PercentileVerdict::Ok(vec![50.5, 99.9]));
    }

    #[test]
    fn boundary_at_100_passes() {
        // p100 = max latency, allowed.
        let v = parse_percentiles("100");
        assert_eq!(v, PercentileVerdict::Ok(vec![100.0]));
    }

    #[test]
    fn zero_rejected() {
        // p0 doesn't make sense (would be min), reject.
        let v = parse_percentiles("0");
        assert!(matches!(v, PercentileVerdict::OutOfBand { .. }));
    }

    #[test]
    fn above_100_rejected() {
        let v = parse_percentiles("150");
        assert!(matches!(v, PercentileVerdict::OutOfBand { .. }));
    }

    #[test]
    fn malformed_token_rejected() {
        let v = parse_percentiles("abc");
        assert!(matches!(v, PercentileVerdict::MalformedToken { .. }));
    }

    #[test]
    fn empty_csv_rejected() {
        assert_eq!(parse_percentiles(""), PercentileVerdict::Empty);
        assert_eq!(parse_percentiles("   "), PercentileVerdict::Empty);
    }
}
