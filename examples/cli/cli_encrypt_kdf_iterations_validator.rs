//! # apr encrypt --kdf-iterations — PBKDF2 Iteration Floor
//!
//! `apr encrypt --passphrase` derives the AES-256 key via PBKDF2-HMAC-
//! SHA256. OWASP 2023 floor: 600,000 iterations; ceiling for usability:
//! 10,000,000 (UI freeze). Below floor → brute-force-feasible; above
//! ceiling → user complaints. This recipe builds the validator + auto-
//! pick by current year.
//!
//! Demonstrates the **ENC.4** recipe for PMAT-115 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ENC-001 + OWASP 2023 PBKDF2 guidelines (RFC 8018)
//!
//! Run with: cargo run --example cli_encrypt_kdf_iterations_validator
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IterVerdict {
    Ok,
    BelowOwaspFloor { recommended: u32 },
    AboveUsabilityCeiling { recommended: u32 },
    InvalidZero,
}

const OWASP_FLOOR_2023: u32 = 600_000;
const USABILITY_CEILING: u32 = 10_000_000;

pub fn classify(iterations: u32) -> IterVerdict {
    if iterations == 0 {
        return IterVerdict::InvalidZero;
    }
    if iterations < OWASP_FLOOR_2023 {
        return IterVerdict::BelowOwaspFloor {
            recommended: OWASP_FLOOR_2023,
        };
    }
    if iterations > USABILITY_CEILING {
        return IterVerdict::AboveUsabilityCeiling {
            recommended: USABILITY_CEILING,
        };
    }
    IterVerdict::Ok
}

pub fn auto_pick(year: u32) -> u32 {
    // Moore's law adaptation: double every 2 years from 2023 baseline.
    if year < 2023 {
        return OWASP_FLOOR_2023;
    }
    let years_elapsed = year - 2023;
    let doublings = years_elapsed / 2;
    let scale = 1u32 << doublings.min(4); // cap at 16x to stay under ceiling
    OWASP_FLOOR_2023.saturating_mul(scale)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_kdf_iterations_validator")?;

    for n in [0u32, 100_000, 600_000, 1_000_000, 20_000_000] {
        println!("iter={n:>10}  →  {:?}", classify(n));
    }
    for y in [2023u32, 2025, 2027, 2030] {
        println!("auto({y}) = {}", auto_pick(y));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(classify(0), IterVerdict::InvalidZero);
    }

    #[test]
    fn legacy_100k_below_floor() {
        let v = classify(100_000);
        assert!(matches!(v, IterVerdict::BelowOwaspFloor { .. }));
    }

    #[test]
    fn at_owasp_floor_passes() {
        assert_eq!(classify(OWASP_FLOOR_2023), IterVerdict::Ok);
    }

    #[test]
    fn between_floor_and_ceiling_ok() {
        assert_eq!(classify(2_000_000), IterVerdict::Ok);
    }

    #[test]
    fn at_usability_ceiling_passes() {
        assert_eq!(classify(USABILITY_CEILING), IterVerdict::Ok);
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = classify(USABILITY_CEILING + 1);
        assert!(matches!(v, IterVerdict::AboveUsabilityCeiling { .. }));
    }

    #[test]
    fn auto_pick_baseline_year_returns_floor() {
        assert_eq!(auto_pick(2023), OWASP_FLOOR_2023);
    }

    #[test]
    fn auto_pick_doubles_every_2_years() {
        assert_eq!(auto_pick(2025), OWASP_FLOOR_2023 * 2);
        assert_eq!(auto_pick(2027), OWASP_FLOOR_2023 * 4);
    }

    #[test]
    fn auto_pick_pre_2023_returns_floor() {
        // Pre-baseline years still get the safe floor.
        assert_eq!(auto_pick(2020), OWASP_FLOOR_2023);
    }

    #[test]
    fn auto_pick_results_always_pass_classify() {
        for y in [2023u32, 2024, 2025, 2026, 2027] {
            let n = auto_pick(y);
            assert_eq!(classify(n), IterVerdict::Ok, "year {y}");
        }
    }
}
