//! # Contracts-Macros Witness Aging Decay
//!
//! Apply exponential decay to witness confidence over time using a
//! configurable half-life. Returns the decayed confidence and whether
//! the witness has expired (below `min_threshold`).
//!
//! Demonstrates the **CMM.171** recipe for PMAT-214 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TLS certificate pinning expiry; OCSP-staple max-age
//!  decay; SLSA attestation freshness windows.
//!
//! Run with: cargo run --example contracts_macros_witness_aging_decay
//!
//! Added by PMAT-214 (catalog 1549→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WitnessAgeVerdict {
    Ok { confidence_x100: u32, expired: bool },
    InvalidConfig,
}

pub fn decay(
    initial_x100: u32,
    age_days: u32,
    half_life_days: u32,
    min_threshold_x100: u32,
) -> WitnessAgeVerdict {
    if initial_x100 == 0 || initial_x100 > 100 || half_life_days == 0 {
        return WitnessAgeVerdict::InvalidConfig;
    }
    let half_lives = age_days as f64 / half_life_days as f64;
    let factor = 0.5f64.powf(half_lives);
    let result = initial_x100 as f64 * factor;
    let result_x100 = result as u32;
    WitnessAgeVerdict::Ok {
        confidence_x100: result_x100,
        expired: result_x100 < min_threshold_x100,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_aging_decay")?;

    println!("fresh: {:?}", decay(95, 0, 90, 50));
    println!("90 days: {:?}", decay(95, 90, 90, 50));
    println!("expired: {:?}", decay(95, 365, 90, 50));
    println!("invalid: {:?}", decay(0, 30, 90, 50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decayer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_age_keeps_full_confidence() {
        let v = decay(95, 0, 90, 50);
        if let WitnessAgeVerdict::Ok {
            confidence_x100, ..
        } = v
        {
            assert_eq!(confidence_x100, 95);
        }
    }

    #[test]
    fn one_half_life_halves_confidence() {
        let v = decay(80, 90, 90, 0);
        if let WitnessAgeVerdict::Ok {
            confidence_x100, ..
        } = v
        {
            assert_eq!(confidence_x100, 40);
        }
    }

    #[test]
    fn invalid_zero_initial() {
        assert_eq!(decay(0, 30, 90, 50), WitnessAgeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_initial_over_100() {
        assert_eq!(decay(101, 30, 90, 50), WitnessAgeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_half_life() {
        assert_eq!(decay(80, 30, 0, 50), WitnessAgeVerdict::InvalidConfig);
    }

    #[test]
    fn old_witness_expires() {
        let v = decay(80, 1000, 90, 50);
        if let WitnessAgeVerdict::Ok { expired, .. } = v {
            assert!(expired);
        }
    }

    #[test]
    fn fresh_witness_not_expired() {
        let v = decay(95, 0, 90, 50);
        if let WitnessAgeVerdict::Ok { expired, .. } = v {
            assert!(!expired);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = decay(80, 30, 90, 50);
        let r2 = decay(80, 30, 90, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn longer_half_life_slower_decay() {
        let fast = decay(80, 90, 30, 0);
        let slow = decay(80, 90, 180, 0);
        if let (
            WitnessAgeVerdict::Ok {
                confidence_x100: f, ..
            },
            WitnessAgeVerdict::Ok {
                confidence_x100: s, ..
            },
        ) = (fast, slow)
        {
            assert!(s > f);
        }
    }

    #[test]
    fn boundary_at_threshold_not_expired() {
        // Exactly at threshold → not expired (uses < not ≤).
        let v = decay(50, 0, 90, 50);
        if let WitnessAgeVerdict::Ok { expired, .. } = v {
            assert!(!expired);
        }
    }

    #[test]
    fn confidence_le_initial() {
        let v = decay(80, 30, 90, 0);
        if let WitnessAgeVerdict::Ok {
            confidence_x100, ..
        } = v
        {
            assert!(confidence_x100 <= 80);
        }
    }

    #[test]
    fn boundary_initial_100() {
        let v = decay(100, 0, 90, 50);
        if let WitnessAgeVerdict::Ok {
            confidence_x100, ..
        } = v
        {
            assert_eq!(confidence_x100, 100);
        }
    }
}
