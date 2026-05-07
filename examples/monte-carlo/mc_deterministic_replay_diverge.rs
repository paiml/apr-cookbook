//! # Monte-Carlo Deterministic Replay Divergence
//!
//! Run two simulation passes with the same seed but slight noise
//! injection; report divergence rate (how often replay diverges
//! despite identical seed). Reproducibility test pattern.
//!
//! Demonstrates the **MC.171** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: rr-debugger record-replay determinism; cf. simular's
//!  `SimRng` at ../aprender/crates/aprender-simulate/src/engine/rng.rs
//!  designed for deterministic replay.
//!
//! Run with: cargo run --example mc_deterministic_replay_diverge
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReplayVerdict {
    Ok {
        diverged: bool,
        first_diverge_step: Option<u32>,
    },
    InvalidConfig,
}

pub fn check(steps: u32, inject_noise_at: Option<u32>, seed: u64) -> ReplayVerdict {
    if steps < 10 {
        return ReplayVerdict::InvalidConfig;
    }
    let mut state_a = seed | 1;
    let mut state_b = seed | 1;
    for i in 0..steps {
        let a = lcg(&mut state_a);
        let mut b = lcg(&mut state_b);
        if inject_noise_at == Some(i) {
            b = b.wrapping_add(1);
        }
        if a != b {
            return ReplayVerdict::Ok {
                diverged: true,
                first_diverge_step: Some(i),
            };
        }
    }
    ReplayVerdict::Ok {
        diverged: false,
        first_diverge_step: None,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_deterministic_replay_diverge")?;

    println!("clean: {:?}", check(100, None, 42));
    println!("noise at 50: {:?}", check(100, Some(50), 42));
    println!("invalid: {:?}", check(5, None, 42));
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
    fn invalid_too_few_steps() {
        assert_eq!(check(5, None, 42), ReplayVerdict::InvalidConfig);
    }

    #[test]
    fn clean_replay_no_divergence() {
        let v = check(100, None, 42);
        assert_eq!(
            v,
            ReplayVerdict::Ok {
                diverged: false,
                first_diverge_step: None,
            }
        );
    }

    #[test]
    fn noise_causes_divergence() {
        let v = check(100, Some(50), 42);
        assert_eq!(
            v,
            ReplayVerdict::Ok {
                diverged: true,
                first_diverge_step: Some(50),
            }
        );
    }

    #[test]
    fn deterministic() {
        let r1 = check(100, None, 42);
        let r2 = check(100, None, 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn noise_at_zero_diverges_first_step() {
        let v = check(100, Some(0), 42);
        assert_eq!(
            v,
            ReplayVerdict::Ok {
                diverged: true,
                first_diverge_step: Some(0),
            }
        );
    }

    #[test]
    fn noise_past_steps_no_diverge() {
        let v = check(100, Some(200), 42);
        assert_eq!(v.is_diverged(), false);
    }

    #[test]
    fn min_steps_accepted() {
        let v = check(10, None, 42);
        assert!(matches!(v, ReplayVerdict::Ok { .. }));
    }

    #[test]
    fn many_steps_handled() {
        let v = check(10_000, None, 42);
        assert!(matches!(v, ReplayVerdict::Ok { .. }));
    }

    #[test]
    fn different_seed_clean_no_divergence() {
        // Both runs use same seed → no internal divergence regardless of seed.
        let v = check(100, None, 999);
        assert_eq!(
            v,
            ReplayVerdict::Ok {
                diverged: false,
                first_diverge_step: None,
            }
        );
    }

    #[test]
    fn noise_at_last_step_diverges() {
        let v = check(100, Some(99), 42);
        if let ReplayVerdict::Ok {
            diverged,
            first_diverge_step,
        } = v
        {
            assert!(diverged);
            assert_eq!(first_diverge_step, Some(99));
        }
    }

    #[test]
    fn noise_in_middle_diverges_at_that_step() {
        let v = check(100, Some(42), 42);
        if let ReplayVerdict::Ok {
            first_diverge_step, ..
        } = v
        {
            assert_eq!(first_diverge_step, Some(42));
        }
    }
}

#[cfg(test)]
impl ReplayVerdict {
    fn is_diverged(&self) -> bool {
        matches!(self, ReplayVerdict::Ok { diverged: true, .. })
    }
}
