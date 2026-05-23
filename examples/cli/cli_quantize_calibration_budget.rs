//! # apr quantize --calibration-samples — Sample Budget Validator
//!
//! Calibration-based quantizers (GPTQ, AWQ, SmoothQuant) need
//! representative samples to fit per-channel scales. Too few → poor
//! generalisation; too many → wall-clock blowup with diminishing
//! returns. Empirical floor: 128 samples; ceiling: 1024 (returns
//! plateau). This recipe builds the budget validator.
//!
//! Demonstrates the **QUANT.5** recipe for PMAT-112 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUANT-001 + Frantar et al. 2023 (GPTQ)
//!
//! Run with: cargo run --example cli_quantize_calibration_budget
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    BelowFloor { recommended: u32 },
    Optimal,
    DiminishingReturns { recommended: u32 },
    InvalidZero,
}

const FLOOR: u32 = 128;
const CEILING: u32 = 1024;

pub fn classify(num_samples: u32) -> BudgetVerdict {
    if num_samples == 0 {
        return BudgetVerdict::InvalidZero;
    }
    if num_samples < FLOOR {
        return BudgetVerdict::BelowFloor { recommended: FLOOR };
    }
    if num_samples > CEILING {
        return BudgetVerdict::DiminishingReturns {
            recommended: CEILING,
        };
    }
    BudgetVerdict::Optimal
}

pub fn estimated_calibration_seconds(num_samples: u32, seq_len: u32) -> u64 {
    // Rough heuristic: 0.1ms/token on a single GPU.
    let total_tokens = u64::from(num_samples) * u64::from(seq_len);
    total_tokens / 10_000
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_calibration_budget")?;

    for n in [0u32, 64, 128, 512, 1024, 2048] {
        let v = classify(n);
        let secs = estimated_calibration_seconds(n, 2048);
        println!("n={n:>4}  →  {v:?}   ~{secs}s @ seq=2048");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(classify(0), BudgetVerdict::InvalidZero);
    }

    #[test]
    fn under_floor_rejected() {
        let v = classify(64);
        assert!(matches!(v, BudgetVerdict::BelowFloor { recommended: 128 }));
    }

    #[test]
    fn at_floor_optimal() {
        assert_eq!(classify(FLOOR), BudgetVerdict::Optimal);
    }

    #[test]
    fn within_band_optimal() {
        assert_eq!(classify(512), BudgetVerdict::Optimal);
    }

    #[test]
    fn at_ceiling_optimal() {
        assert_eq!(classify(CEILING), BudgetVerdict::Optimal);
    }

    #[test]
    fn over_ceiling_diminishing_returns() {
        let v = classify(2048);
        assert!(matches!(
            v,
            BudgetVerdict::DiminishingReturns { recommended: 1024 }
        ));
    }

    #[test]
    fn calibration_time_scales_with_samples() {
        let t1 = estimated_calibration_seconds(128, 2048);
        let t2 = estimated_calibration_seconds(512, 2048);
        assert!(t2 > t1);
    }

    #[test]
    fn calibration_time_scales_with_seq_len() {
        let t_short = estimated_calibration_seconds(128, 512);
        let t_long = estimated_calibration_seconds(128, 4096);
        assert!(t_long > t_short);
    }
}
