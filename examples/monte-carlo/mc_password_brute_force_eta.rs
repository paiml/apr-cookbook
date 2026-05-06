//! # Monte-Carlo Password Brute-Force ETA
//!
//! Estimate brute-force time for a uniform-random password drawn
//! from `alphabet_size^length`. Each trial does N attempts/sec with
//! jitter; reports mean ETA in seconds.
//!
//! Demonstrates the **MC.87** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NIST SP 800-63B Appendix A; brute-force entropy
//!  estimation conventions.
//!
//! Run with: cargo run --example mc_password_brute_force_eta
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EtaVerdict {
    Ok {
        mean_eta_secs: f64,
        worst_case_secs: u64,
    },
    InvalidConfig,
}

pub fn simulate(
    trials: u32,
    alphabet_size: u32,
    length: u32,
    attempts_per_sec: u32,
    seed: u64,
) -> EtaVerdict {
    if trials == 0 || alphabet_size < 2 || length == 0 || attempts_per_sec == 0 {
        return EtaVerdict::InvalidConfig;
    }
    let space: u64 = (alphabet_size as u64).saturating_pow(length);
    let mut rng_state = seed | 1;
    let mut total_secs: f64 = 0.0;
    for _ in 0..trials {
        let target_pos = (lcg(&mut rng_state) % space) + 1;
        let secs = target_pos as f64 / f64::from(attempts_per_sec);
        total_secs += secs;
    }
    EtaVerdict::Ok {
        mean_eta_secs: total_secs / f64::from(trials),
        worst_case_secs: space / u64::from(attempts_per_sec),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_password_brute_force_eta")?;

    println!("4-digit pin: {:?}", simulate(1000, 10, 4, 1000, 42));
    println!("8-char alpha: {:?}", simulate(100, 26, 8, 1_000_000, 42));
    println!("invalid: {:?}", simulate(0, 10, 4, 1000, 42));
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
    fn worst_case_le_full_search() {
        // 10^4 = 10_000 / 1000 attempts/s = 10 seconds.
        let v = simulate(100, 10, 4, 1000, 42);
        if let EtaVerdict::Ok {
            worst_case_secs, ..
        } = v
        {
            assert_eq!(worst_case_secs, 10);
        }
    }

    #[test]
    fn mean_le_worst_case() {
        let v = simulate(1000, 10, 4, 1000, 42);
        if let EtaVerdict::Ok {
            mean_eta_secs,
            worst_case_secs,
        } = v
        {
            assert!(mean_eta_secs <= worst_case_secs as f64);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 4, 1000, 42), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_alphabet_too_small() {
        assert_eq!(simulate(100, 1, 4, 1000, 42), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_length() {
        assert_eq!(simulate(100, 10, 0, 1000, 42), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_attempts_per_sec() {
        assert_eq!(simulate(100, 10, 4, 0, 42), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 4, 1000, 42);
        let b = simulate(500, 10, 4, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_around_half_worst() {
        // For uniform distribution, mean position ≈ N/2 → mean ETA ≈ worst/2.
        let v = simulate(5000, 10, 4, 1000, 42);
        if let EtaVerdict::Ok {
            mean_eta_secs,
            worst_case_secs,
        } = v
        {
            let half = worst_case_secs as f64 / 2.0;
            assert!((mean_eta_secs - half).abs() / half < 0.20);
        }
    }

    #[test]
    fn longer_password_higher_eta() {
        let short = simulate(100, 10, 4, 1000, 42);
        let long = simulate(100, 10, 8, 1000, 42);
        if let (
            EtaVerdict::Ok {
                worst_case_secs: s, ..
            },
            EtaVerdict::Ok {
                worst_case_secs: l, ..
            },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn larger_alphabet_higher_eta() {
        let small = simulate(100, 10, 4, 1000, 42);
        let big = simulate(100, 26, 4, 1000, 42);
        if let (
            EtaVerdict::Ok {
                worst_case_secs: s, ..
            },
            EtaVerdict::Ok {
                worst_case_secs: b, ..
            },
        ) = (small, big)
        {
            assert!(b > s);
        }
    }

    #[test]
    fn mean_eta_nonneg() {
        let v = simulate(100, 10, 4, 1000, 42);
        if let EtaVerdict::Ok { mean_eta_secs, .. } = v {
            assert!(mean_eta_secs >= 0.0);
        }
    }
}
