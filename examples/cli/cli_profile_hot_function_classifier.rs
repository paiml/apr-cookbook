//! # apr profile — Hot Function Classifier
//!
//! `apr profile` reports per-function self-time percentage. Tiers:
//! < 1% = noise, 1-10% = warm, 10-30% = hot, > 30% = bottleneck.
//! This recipe builds the classifier + Pareto-cumulative reporter
//! (the 80/20 rule applied to perf optimization).
//!
//! Demonstrates the **PROF.6** recipe for PMAT-115 (apr profile coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PROF-001 + Pareto principle (Juran 1941)
//!
//! Run with: cargo run --example cli_profile_hot_function_classifier
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HotnessTier {
    Noise,
    Warm,
    Hot,
    Bottleneck,
    InvalidPct,
}

pub fn classify(self_pct: f64) -> HotnessTier {
    if !self_pct.is_finite() || !(0.0..=100.0).contains(&self_pct) {
        return HotnessTier::InvalidPct;
    }
    if self_pct < 1.0 {
        HotnessTier::Noise
    } else if self_pct < 10.0 {
        HotnessTier::Warm
    } else if self_pct < 30.0 {
        HotnessTier::Hot
    } else {
        HotnessTier::Bottleneck
    }
}

pub fn pareto_top_count(sorted_pcts: &[f64], cumulative_target: f64) -> usize {
    let mut sum = 0.0;
    for (i, p) in sorted_pcts.iter().enumerate() {
        sum += p;
        if sum >= cumulative_target {
            return i + 1;
        }
    }
    sorted_pcts.len()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_profile_hot_function_classifier")?;

    let pcts = [42.0, 18.0, 12.0, 8.0, 5.0, 3.0, 1.5, 0.5];
    for p in pcts {
        println!("{p:>5.1}%  →  {:?}", classify(p));
    }
    println!("Pareto-80: top {}", pareto_top_count(&pcts, 80.0));
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
    fn under_1pct_noise() {
        assert_eq!(classify(0.5), HotnessTier::Noise);
        assert_eq!(classify(0.99), HotnessTier::Noise);
    }

    #[test]
    fn one_to_10_warm() {
        assert_eq!(classify(1.0), HotnessTier::Warm);
        assert_eq!(classify(9.99), HotnessTier::Warm);
    }

    #[test]
    fn ten_to_30_hot() {
        assert_eq!(classify(10.0), HotnessTier::Hot);
        assert_eq!(classify(29.99), HotnessTier::Hot);
    }

    #[test]
    fn over_30_bottleneck() {
        assert_eq!(classify(30.0), HotnessTier::Bottleneck);
        assert_eq!(classify(95.0), HotnessTier::Bottleneck);
    }

    #[test]
    fn negative_or_over_100_invalid() {
        assert_eq!(classify(-0.1), HotnessTier::InvalidPct);
        assert_eq!(classify(100.01), HotnessTier::InvalidPct);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(classify(f64::NAN), HotnessTier::InvalidPct);
    }

    #[test]
    fn pareto_finds_top_n_for_threshold() {
        // 42 + 18 + 12 + 8 = 80 → 4 functions cover 80%.
        let pcts = [42.0, 18.0, 12.0, 8.0, 5.0, 3.0];
        assert_eq!(pareto_top_count(&pcts, 80.0), 4);
    }

    #[test]
    fn pareto_returns_full_len_if_target_unreachable() {
        let pcts = [10.0, 10.0];
        assert_eq!(pareto_top_count(&pcts, 80.0), 2);
    }

    #[test]
    fn pareto_first_function_alone_covers_threshold() {
        let pcts = [95.0, 5.0];
        assert_eq!(pareto_top_count(&pcts, 80.0), 1);
    }
}
