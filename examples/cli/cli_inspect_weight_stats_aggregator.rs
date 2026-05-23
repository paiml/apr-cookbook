//! # apr inspect --weights — Per-Tensor Weight Stats Aggregator
//!
//! `apr inspect --weights <FILE>` shows per-tensor (mean, std, min,
//! max, n_zero, n_inf, n_nan) plus aggregate health flags. This recipe
//! builds the aggregator and asserts the contract: NaN/inf counts must
//! be 0 for a healthy model; zero-fraction > 0.5 indicates dead neurons.
//!
//! Demonstrates the **INSPECT.7** recipe for PMAT-109 (apr inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender INSPECT-002
//!
//! Run with: cargo run --example cli_inspect_weight_stats_aggregator
//!
//! Added by PMAT-109 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct WeightStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub n_zero: u64,
    pub n_inf: u64,
    pub n_nan: u64,
    pub n_total: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealthFlag {
    HasNaN,
    HasInf,
    DeadNeurons { fraction: f64 },
    SaturatedRange,
}

const DEAD_FRACTION_THRESHOLD: f64 = 0.5;

pub fn compute_stats(values: &[f64]) -> WeightStats {
    let n_total = values.len() as u64;
    if n_total == 0 {
        return WeightStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            n_zero: 0,
            n_inf: 0,
            n_nan: 0,
            n_total: 0,
        };
    }
    let n_nan = values.iter().filter(|v| v.is_nan()).count() as u64;
    let n_inf = values.iter().filter(|v| v.is_infinite()).count() as u64;
    let n_zero = values.iter().filter(|v| **v == 0.0).count() as u64;
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return WeightStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            n_zero,
            n_inf,
            n_nan,
            n_total,
        };
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let std = var.sqrt();
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    WeightStats {
        mean,
        std,
        min,
        max,
        n_zero,
        n_inf,
        n_nan,
        n_total,
    }
}

pub fn health_flags(s: &WeightStats) -> Vec<HealthFlag> {
    let mut flags = Vec::new();
    if s.n_nan > 0 {
        flags.push(HealthFlag::HasNaN);
    }
    if s.n_inf > 0 {
        flags.push(HealthFlag::HasInf);
    }
    if s.n_total > 0 {
        let zero_frac = s.n_zero as f64 / s.n_total as f64;
        if zero_frac > DEAD_FRACTION_THRESHOLD {
            flags.push(HealthFlag::DeadNeurons {
                fraction: zero_frac,
            });
        }
    }
    if s.min == s.max && s.n_total > 1 {
        flags.push(HealthFlag::SaturatedRange);
    }
    flags
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_inspect_weight_stats_aggregator")?;

    let healthy = [0.1, 0.2, -0.3, 0.05, -0.15, 0.4];
    let dead = [0.0_f64; 99]; // all zero
    let nan_inf = [0.1, f64::NAN, f64::INFINITY, 0.5];

    for (label, vals) in [
        ("healthy", &healthy[..]),
        ("dead", &dead[..]),
        ("nan/inf", &nan_inf[..]),
    ] {
        let s = compute_stats(vals);
        println!("{label:>10}: stats={s:?}");
        println!("  flags: {:?}", health_flags(&s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_returns_zeros() {
        let s = compute_stats(&[]);
        assert_eq!(s.n_total, 0);
        assert_eq!(s.n_nan, 0);
    }

    #[test]
    fn nan_counted_separately() {
        let s = compute_stats(&[1.0, f64::NAN, 2.0]);
        assert_eq!(s.n_nan, 1);
        assert_eq!(s.n_inf, 0);
    }

    #[test]
    fn inf_counted_separately() {
        let s = compute_stats(&[1.0, f64::INFINITY, 2.0]);
        assert_eq!(s.n_inf, 1);
        assert_eq!(s.n_nan, 0);
    }

    #[test]
    fn zero_count_correct() {
        let s = compute_stats(&[0.0, 1.0, 0.0, 2.0, 0.0]);
        assert_eq!(s.n_zero, 3);
    }

    #[test]
    fn healthy_has_no_flags() {
        let s = compute_stats(&[0.1, 0.2, -0.3, 0.4]);
        assert!(health_flags(&s).is_empty());
    }

    #[test]
    fn nan_triggers_flag() {
        let s = compute_stats(&[0.1, f64::NAN]);
        assert!(health_flags(&s).contains(&HealthFlag::HasNaN));
    }

    #[test]
    fn dead_neurons_above_50pct_triggers_flag() {
        let s = compute_stats(&[0.0, 0.0, 0.0, 0.1, 0.2]);
        let flags = health_flags(&s);
        assert!(flags
            .iter()
            .any(|f| matches!(f, HealthFlag::DeadNeurons { .. })));
    }

    #[test]
    fn dead_neurons_below_50pct_does_not_trigger() {
        let s = compute_stats(&[0.0, 0.1, 0.2, 0.3, 0.4]);
        let flags = health_flags(&s);
        assert!(!flags
            .iter()
            .any(|f| matches!(f, HealthFlag::DeadNeurons { .. })));
    }
}
