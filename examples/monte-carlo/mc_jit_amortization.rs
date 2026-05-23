//! # Monte-Carlo JIT Amortization
//!
//! Sim JIT-compilation amortization. First N requests pay
//! compile-time penalty; subsequent requests pay only run-time.
//! Returns the break-even request number where total cost falls
//! below the always-AOT cost.
//!
//! Demonstrates the **MC.60** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: V8 / GraalVM tiered-compilation amortization studies.
//!
//! Run with: cargo run --example mc_jit_amortization
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AmortizeVerdict {
    Ok {
        breakeven_request: Option<u32>,
        jit_total_ms: f64,
        aot_total_ms: f64,
    },
    InvalidConfig,
}

pub fn analyze(
    aot_per_request_ms: f64,
    jit_compile_ms: f64,
    jit_per_request_ms: f64,
    requests: u32,
) -> AmortizeVerdict {
    if !aot_per_request_ms.is_finite()
        || aot_per_request_ms <= 0.0
        || !jit_compile_ms.is_finite()
        || jit_compile_ms < 0.0
        || !jit_per_request_ms.is_finite()
        || jit_per_request_ms <= 0.0
        || jit_per_request_ms > aot_per_request_ms
        || requests == 0
    {
        return AmortizeVerdict::InvalidConfig;
    }
    let mut breakeven_request: Option<u32> = None;
    for r in 1..=requests {
        let jit_total = jit_compile_ms + f64::from(r) * jit_per_request_ms;
        let aot_total = f64::from(r) * aot_per_request_ms;
        if jit_total <= aot_total && breakeven_request.is_none() {
            breakeven_request = Some(r);
        }
    }
    let jit_total_ms = jit_compile_ms + f64::from(requests) * jit_per_request_ms;
    let aot_total_ms = f64::from(requests) * aot_per_request_ms;
    AmortizeVerdict::Ok {
        breakeven_request,
        jit_total_ms,
        aot_total_ms,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_jit_amortization")?;

    println!("typical: {:?}", analyze(10.0, 100.0, 5.0, 1000));
    println!("no breakeven: {:?}", analyze(10.0, 100.0, 9.0, 5));
    println!("invalid: {:?}", analyze(10.0, 100.0, 12.0, 1000));
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
    fn typical_breakeven() {
        let v = analyze(10.0, 100.0, 5.0, 1000);
        if let AmortizeVerdict::Ok {
            breakeven_request, ..
        } = v
        {
            // breakeven: 100 + r*5 ≤ 10*r → r ≥ 20.
            assert_eq!(breakeven_request, Some(20));
        }
    }

    #[test]
    fn no_breakeven_short_run() {
        let v = analyze(10.0, 1000.0, 5.0, 5);
        if let AmortizeVerdict::Ok {
            breakeven_request, ..
        } = v
        {
            assert_eq!(breakeven_request, None);
        }
    }

    #[test]
    fn jit_below_aot_at_breakeven() {
        let v = analyze(10.0, 100.0, 5.0, 1000);
        if let AmortizeVerdict::Ok {
            jit_total_ms,
            aot_total_ms,
            ..
        } = v
        {
            assert!(jit_total_ms < aot_total_ms);
        }
    }

    #[test]
    fn invalid_jit_above_aot() {
        assert_eq!(
            analyze(10.0, 100.0, 12.0, 1000),
            AmortizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_aot() {
        assert_eq!(
            analyze(0.0, 100.0, 5.0, 1000),
            AmortizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_compile() {
        assert_eq!(
            analyze(10.0, -1.0, 5.0, 1000),
            AmortizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(analyze(10.0, 100.0, 5.0, 0), AmortizeVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            analyze(f64::NAN, 100.0, 5.0, 1000),
            AmortizeVerdict::InvalidConfig
        );
    }

    #[test]
    fn no_compile_breakeven_immediate() {
        let v = analyze(10.0, 0.0, 5.0, 100);
        if let AmortizeVerdict::Ok {
            breakeven_request, ..
        } = v
        {
            assert_eq!(breakeven_request, Some(1));
        }
    }

    #[test]
    fn deterministic() {
        let a = analyze(10.0, 100.0, 5.0, 1000);
        let b = analyze(10.0, 100.0, 5.0, 1000);
        assert_eq!(a, b);
    }
}
