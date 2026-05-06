//! # Monte-Carlo Speculative-Decode Acceptance Rate
//!
//! Sim speculative-decoding: a draft model proposes K tokens; the
//! verifier accepts each with `accept_prob`. First rejection truncates
//! the speculation. Returns observed mean accepted / sample.
//!
//! Demonstrates the **MC.46** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Leviathan et al. (2023). Speculative Decoding.
//!
//! Run with: cargo run --example mc_speculative_decode_acceptance
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SpeculativeVerdict {
    Ok {
        mean_accepted: f64,
        full_acceptance_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    speculative_window: u32,
    accept_prob: f64,
    samples: u32,
    seed: u64,
) -> SpeculativeVerdict {
    if speculative_window == 0
        || samples == 0
        || !accept_prob.is_finite()
        || !(0.0..=1.0).contains(&accept_prob)
    {
        return SpeculativeVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_accepted: u64 = 0;
    let mut full_runs: u32 = 0;
    for _ in 0..samples {
        let mut accepted: u32 = 0;
        for _ in 0..speculative_window {
            if unit(&mut rng_state) < accept_prob {
                accepted += 1;
            } else {
                break;
            }
        }
        total_accepted += u64::from(accepted);
        if accepted == speculative_window {
            full_runs += 1;
        }
    }
    let mean_accepted = total_accepted as f64 / f64::from(samples);
    let full_acceptance_rate = f64::from(full_runs) / f64::from(samples);
    SpeculativeVerdict::Ok {
        mean_accepted,
        full_acceptance_rate,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_speculative_decode_acceptance")?;

    println!("high quality: {:?}", simulate(8, 0.95, 1000, 42));
    println!("low quality: {:?}", simulate(8, 0.50, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.95, 1000, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_prob_high_mean() {
        let v = simulate(8, 0.95, 10_000, 42);
        if let SpeculativeVerdict::Ok { mean_accepted, .. } = v {
            assert!(mean_accepted > 6.0);
        }
    }

    #[test]
    fn low_prob_low_mean() {
        let v = simulate(8, 0.20, 10_000, 42);
        if let SpeculativeVerdict::Ok { mean_accepted, .. } = v {
            assert!(mean_accepted < 1.0);
        }
    }

    #[test]
    fn p_one_full_window() {
        let v = simulate(8, 1.0, 100, 42);
        if let SpeculativeVerdict::Ok {
            mean_accepted,
            full_acceptance_rate,
        } = v
        {
            assert!((mean_accepted - 8.0).abs() < 1e-9);
            assert!((full_acceptance_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn p_zero_no_acceptance() {
        let v = simulate(8, 0.0, 100, 42);
        if let SpeculativeVerdict::Ok { mean_accepted, .. } = v {
            assert!((mean_accepted - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(simulate(0, 0.5, 100, 42), SpeculativeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(8, 0.5, 0, 42), SpeculativeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_prob() {
        assert_eq!(
            simulate(8, -0.1, 100, 42),
            SpeculativeVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(8, f64::NAN, 100, 42),
            SpeculativeVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(8, 0.7, 1000, 42);
        let b = simulate(8, 0.7, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_window_lower_full_rate() {
        let small = simulate(2, 0.7, 5000, 42);
        let large = simulate(20, 0.7, 5000, 42);
        if let (
            SpeculativeVerdict::Ok {
                full_acceptance_rate: s,
                ..
            },
            SpeculativeVerdict::Ok {
                full_acceptance_rate: l,
                ..
            },
        ) = (small, large)
        {
            assert!(s > l);
        }
    }
}
