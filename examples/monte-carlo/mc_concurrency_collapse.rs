//! # Monte-Carlo Universal Scalability Law (USL) Concurrency Collapse
//!
//! Apply Gunther's USL: throughput = N / (1 + alpha (N-1) + beta N(N-1)).
//! Returns the concurrency level beyond which throughput collapses.
//!
//! Demonstrates the **MC.50** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gunther's Universal Scalability Law (1993).
//!
//! Run with: cargo run --example mc_concurrency_collapse
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CollapseVerdict {
    Ok {
        peak_concurrency: u32,
        peak_throughput: f64,
        throughput_at_2x_peak: f64,
    },
    InvalidConfig,
}

pub fn analyze(alpha: f64, beta: f64, max_concurrency: u32) -> CollapseVerdict {
    if !alpha.is_finite() || alpha < 0.0 || !beta.is_finite() || beta < 0.0 || max_concurrency < 2 {
        return CollapseVerdict::InvalidConfig;
    }
    let mut peak_n = 1u32;
    let mut peak_x = 0.0_f64;
    for n in 1..=max_concurrency {
        let nf = f64::from(n);
        let denom = 1.0 + alpha * (nf - 1.0) + beta * nf * (nf - 1.0);
        let x = nf / denom;
        if x > peak_x {
            peak_x = x;
            peak_n = n;
        }
    }
    let n2 = peak_n.saturating_mul(2).min(max_concurrency);
    let n2f = f64::from(n2);
    let denom2 = 1.0 + alpha * (n2f - 1.0) + beta * n2f * (n2f - 1.0);
    let throughput_at_2x_peak = n2f / denom2;
    CollapseVerdict::Ok {
        peak_concurrency: peak_n,
        peak_throughput: peak_x,
        throughput_at_2x_peak,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_concurrency_collapse")?;

    println!("contention only: {:?}", analyze(0.05, 0.0, 200));
    println!("with crosstalk: {:?}", analyze(0.05, 0.001, 200));
    println!("invalid: {:?}", analyze(-0.1, 0.0, 200));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyzer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pure_contention_amdahl_limit() {
        // No crosstalk → throughput → 1/alpha as N → ∞ (Amdahl).
        let v = analyze(0.05, 0.0, 1000);
        if let CollapseVerdict::Ok {
            peak_concurrency, ..
        } = v
        {
            // Throughput is monotonically increasing → peak at max_concurrency.
            assert_eq!(peak_concurrency, 1000);
        }
    }

    #[test]
    fn crosstalk_causes_finite_peak() {
        let v = analyze(0.05, 0.001, 1000);
        if let CollapseVerdict::Ok {
            peak_concurrency, ..
        } = v
        {
            // Peak should be well below max.
            assert!(peak_concurrency < 100);
        }
    }

    #[test]
    fn beyond_peak_throughput_drops() {
        let v = analyze(0.05, 0.001, 1000);
        if let CollapseVerdict::Ok {
            peak_throughput,
            throughput_at_2x_peak,
            ..
        } = v
        {
            assert!(throughput_at_2x_peak < peak_throughput);
        }
    }

    #[test]
    fn invalid_neg_alpha() {
        assert_eq!(analyze(-0.1, 0.0, 100), CollapseVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_beta() {
        assert_eq!(analyze(0.05, -0.1, 100), CollapseVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_low_max() {
        assert_eq!(analyze(0.05, 0.001, 1), CollapseVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            analyze(f64::NAN, 0.001, 100),
            CollapseVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = analyze(0.05, 0.001, 100);
        let b = analyze(0.05, 0.001, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn no_overhead_linear() {
        // alpha=beta=0 → throughput = N. Peak at max.
        let v = analyze(0.0, 0.0, 100);
        if let CollapseVerdict::Ok {
            peak_concurrency,
            peak_throughput,
            ..
        } = v
        {
            assert_eq!(peak_concurrency, 100);
            assert!((peak_throughput - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn higher_beta_earlier_peak() {
        let lo = analyze(0.05, 0.0001, 1000);
        let hi = analyze(0.05, 0.01, 1000);
        if let (
            CollapseVerdict::Ok {
                peak_concurrency: l,
                ..
            },
            CollapseVerdict::Ok {
                peak_concurrency: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h < l);
        }
    }
}
