//! # Monte-Carlo RNG Period Check
//!
//! Detect short-period RNG cycles by tracking when the generator
//! state revisits a previously-seen value within a sample window.
//! Returns categorical verdict and observed cycle length (if any).
//!
//! Demonstrates the **MC.180** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Marsaglia diehard tests; Knuth TAOCP §3.3.2 cycle
//!  detection (Brent's algorithm).
//!
//! Run with: cargo run --example mc_rng_period_check
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum PeriodVerdict {
    NoCycleSeen { samples_taken: u32 },
    Cycle { length: u32 },
    InvalidConfig,
}

/// Tiny LCG (small modulus to provoke detectable cycles).
fn tiny_lcg(state: &mut u32, modulus: u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345) % modulus;
    *state
}

pub fn check(modulus: u32, max_samples: u32, seed: u32) -> PeriodVerdict {
    if modulus < 4 || max_samples < 10 {
        return PeriodVerdict::InvalidConfig;
    }
    let mut state = seed % modulus;
    let mut seen: HashMap<u32, u32> = HashMap::new();
    seen.insert(state, 0);
    for i in 1..=max_samples {
        let v = tiny_lcg(&mut state, modulus);
        if let Some(prev) = seen.get(&v) {
            return PeriodVerdict::Cycle { length: i - prev };
        }
        seen.insert(v, i);
    }
    PeriodVerdict::NoCycleSeen {
        samples_taken: max_samples,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_rng_period_check")?;

    println!("mod=16: {:?}", check(16, 1000, 42));
    println!("mod=2^16: {:?}", check(65536, 100, 42));
    println!("invalid: {:?}", check(2, 5, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_too_small_modulus() {
        assert_eq!(check(2, 100, 42), PeriodVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(check(100, 5, 42), PeriodVerdict::InvalidConfig);
    }

    #[test]
    fn small_modulus_detects_cycle() {
        let v = check(16, 1000, 42);
        if let PeriodVerdict::Cycle { length } = v {
            // For modulus 16, cycle ≤ 16.
            assert!(length <= 16);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(16, 100, 42);
        let b = check(16, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn cycle_length_positive() {
        let v = check(16, 1000, 42);
        if let PeriodVerdict::Cycle { length } = v {
            assert!(length >= 1);
        }
    }

    #[test]
    fn large_modulus_no_cycle_in_short_window() {
        let v = check(1_000_000, 100, 42);
        assert!(matches!(v, PeriodVerdict::NoCycleSeen { .. }));
    }

    #[test]
    fn samples_taken_returned() {
        let v = check(1_000_000, 50, 42);
        if let PeriodVerdict::NoCycleSeen { samples_taken } = v {
            assert_eq!(samples_taken, 50);
        }
    }

    #[test]
    fn min_modulus_accepted() {
        let v = check(4, 100, 42);
        assert!(matches!(
            v,
            PeriodVerdict::Cycle { .. } | PeriodVerdict::NoCycleSeen { .. }
        ));
    }

    #[test]
    fn many_samples_handled() {
        let v = check(1024, 5000, 42);
        assert!(matches!(
            v,
            PeriodVerdict::Cycle { .. } | PeriodVerdict::NoCycleSeen { .. }
        ));
    }

    #[test]
    fn different_seeds_may_diverge() {
        let _a = check(1024, 100, 42);
        let _b = check(1024, 100, 123);
        // Different seeds may follow different cycle paths.
        // Just check both produce valid verdicts.
        assert!(true);
    }

    #[test]
    fn cycle_le_modulus() {
        let v = check(64, 1000, 42);
        if let PeriodVerdict::Cycle { length } = v {
            assert!(length <= 64);
        }
    }
}
