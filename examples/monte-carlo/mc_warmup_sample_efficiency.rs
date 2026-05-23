//! # Monte-Carlo Warmup Sample Efficiency
//!
//! Estimate effective sample size (ESS) during warmup. Correlated
//! samples have low ESS; warmup discards burn-in to improve ESS.
//! Returns ESS / N for the warmup window.
//!
//! Demonstrates the **MC.34** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: MCMC effective sample size (Geyer 1992).
//!
//! Run with: cargo run --example mc_warmup_sample_efficiency
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EssVerdict {
    Ok {
        effective_size: f64,
        efficiency_pct: f64,
    },
    InvalidConfig,
}

pub fn estimate(total_samples: u32, autocorrelation: f64, warmup_discarded: u32) -> EssVerdict {
    if total_samples == 0
        || warmup_discarded > total_samples
        || !autocorrelation.is_finite()
        || autocorrelation.abs() >= 1.0
        || autocorrelation < 0.0
    {
        return EssVerdict::InvalidConfig;
    }
    let kept = f64::from(total_samples - warmup_discarded);
    // Standard MCMC formula: ESS = N * (1 - rho) / (1 + rho).
    let factor = (1.0 - autocorrelation) / (1.0 + autocorrelation);
    let effective_size = kept * factor;
    let efficiency_pct = if kept > 0.0 {
        (effective_size / kept) * 100.0
    } else {
        0.0
    };
    EssVerdict::Ok {
        effective_size,
        efficiency_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_warmup_sample_efficiency")?;

    println!("low corr: {:?}", estimate(1000, 0.1, 100));
    println!("high corr: {:?}", estimate(1000, 0.9, 100));
    println!("no warmup: {:?}", estimate(1000, 0.5, 0));
    println!("invalid: {:?}", estimate(0, 0.5, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn low_correlation_high_efficiency() {
        let v = estimate(1000, 0.1, 0);
        if let EssVerdict::Ok { efficiency_pct, .. } = v {
            assert!(efficiency_pct > 70.0);
        }
    }

    #[test]
    fn high_correlation_low_efficiency() {
        let v = estimate(1000, 0.9, 0);
        if let EssVerdict::Ok { efficiency_pct, .. } = v {
            assert!(efficiency_pct < 10.0);
        }
    }

    #[test]
    fn warmup_discards_samples() {
        let v_no = estimate(1000, 0.5, 0);
        let v_warm = estimate(1000, 0.5, 200);
        if let (
            EssVerdict::Ok {
                effective_size: a, ..
            },
            EssVerdict::Ok {
                effective_size: b, ..
            },
        ) = (v_no, v_warm)
        {
            assert!(a > b);
        }
    }

    #[test]
    fn zero_correlation_full_efficiency() {
        let v = estimate(1000, 0.0, 0);
        if let EssVerdict::Ok { efficiency_pct, .. } = v {
            assert!((efficiency_pct - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(estimate(0, 0.5, 0), EssVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_warmup_over_total() {
        assert_eq!(estimate(100, 0.5, 200), EssVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_corr() {
        assert_eq!(estimate(100, -0.5, 0), EssVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_corr_one() {
        assert_eq!(estimate(100, 1.0, 0), EssVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(estimate(100, f64::NAN, 0), EssVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(1000, 0.5, 100);
        let b = estimate(1000, 0.5, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn ess_bounded_by_kept() {
        let v = estimate(1000, 0.5, 100);
        if let EssVerdict::Ok { effective_size, .. } = v {
            assert!(effective_size <= 900.0);
        }
    }
}
