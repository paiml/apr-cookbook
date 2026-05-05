//! # apr diagnose — Model Size Inference Heuristic
//!
//! `apr diagnose <DIR> --model-size <SIZE>` accepts a free-form size hint
//! (`"0.5B"`, `"7b"`, `"tiny"`, `"125M"`) and the binary normalizes it
//! into an internal `(parameter_count, family_class)` tuple. This recipe
//! exposes the parser so a CI pipeline can preview how a hint will be
//! interpreted before the binary picks profiling defaults from it.
//!
//! Demonstrates the **DIAGNOSE.5** recipe for PMAT-095 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAGNOSE-003
//!
//! Run with: cargo run --example cli_diagnose_model_size_inference
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum SizeClass {
    Tiny,     // < 100M params
    Small,    // 100M..1B
    Medium,   // 1B..10B
    Large,    // 10B..100B
    Frontier, // ≥ 100B
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSize {
    pub raw: String,
    pub params: u64,
    pub class: SizeClass,
}

pub fn parse_size_hint(hint: &str) -> Option<ParsedSize> {
    let lower = hint.trim().to_ascii_lowercase();

    // Named buckets first.
    let named = match lower.as_str() {
        "tiny" => Some(50_000_000u64),
        "small" => Some(500_000_000),
        "medium" | "med" => Some(7_000_000_000),
        "large" => Some(70_000_000_000),
        "frontier" | "xl" => Some(400_000_000_000),
        _ => None,
    };
    if let Some(p) = named {
        return Some(ParsedSize {
            raw: hint.into(),
            params: p,
            class: classify(p),
        });
    }

    // Numeric form: e.g. "0.5B", "7b", "125M".
    let last = lower.chars().last()?;
    let multiplier: u64 = match last {
        'k' => 1_000,
        'm' => 1_000_000,
        'b' => 1_000_000_000,
        't' => 1_000_000_000_000,
        _ => return None,
    };
    let num_part: String = lower
        .chars()
        .take(lower.len() - 1)
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let n: f64 = num_part.parse().ok()?;
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    let params = (n * multiplier as f64) as u64;
    Some(ParsedSize {
        raw: hint.into(),
        params,
        class: classify(params),
    })
}

fn classify(params: u64) -> SizeClass {
    match params {
        n if n < 100_000_000 => SizeClass::Tiny,
        n if n < 1_000_000_000 => SizeClass::Small,
        n if n < 10_000_000_000 => SizeClass::Medium,
        n if n < 100_000_000_000 => SizeClass::Large,
        _ => SizeClass::Frontier,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_model_size_inference")?;

    for hint in [
        "tiny", "0.5B", "7b", "125M", "70B", "400B", "garbage", "1.5T",
    ] {
        println!("{hint:>10} → {:?}", parse_size_hint(hint));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn named_tiny_classifies_as_tiny() {
        let p = parse_size_hint("tiny").unwrap();
        assert_eq!(p.class, SizeClass::Tiny);
    }

    #[test]
    fn numeric_500m_is_small() {
        let p = parse_size_hint("500M").unwrap();
        assert_eq!(p.params, 500_000_000);
        assert_eq!(p.class, SizeClass::Small);
    }

    #[test]
    fn numeric_7b_is_medium() {
        let p = parse_size_hint("7B").unwrap();
        assert_eq!(p.params, 7_000_000_000);
        assert_eq!(p.class, SizeClass::Medium);
    }

    #[test]
    fn fractional_size_handled() {
        // "0.5B" → 500M (fractional is common for sub-1B models).
        let p = parse_size_hint("0.5B").unwrap();
        assert_eq!(p.params, 500_000_000);
    }

    #[test]
    fn case_insensitive() {
        let lower = parse_size_hint("7b").unwrap();
        let upper = parse_size_hint("7B").unwrap();
        assert_eq!(lower.params, upper.params);
    }

    #[test]
    fn garbage_returns_none_not_zero() {
        // Garbage MUST return None — a silent 0 would pick wrong defaults.
        assert!(parse_size_hint("garbage").is_none());
        assert!(parse_size_hint("").is_none());
        assert!(parse_size_hint("xyz").is_none());
    }

    #[test]
    fn frontier_size_classifies() {
        let p = parse_size_hint("400B").unwrap();
        assert_eq!(p.class, SizeClass::Frontier);
    }
}
