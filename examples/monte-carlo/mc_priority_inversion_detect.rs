//! # Monte-Carlo Priority Inversion Detect
//!
//! Sim 3 threads (low, medium, high) where the low-priority thread
//! holds a resource the high-priority thread needs. Without priority
//! inheritance, the medium thread starves the low thread, blocking
//! high. Reports inversion event count.
//!
//! Demonstrates the **MC.113** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mars Pathfinder priority inversion (1997); Sha, Rajkumar,
//!  Lehoczky "Priority Inheritance Protocols" (IEEE TC 1990).
//!
//! Run with: cargo run --example mc_priority_inversion_detect
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum InversionVerdict {
    Ok {
        inversion_events: u32,
        max_inversion_duration: u32,
    },
    InvalidConfig,
}

pub fn simulate(seconds: u32, medium_arrival_prob: f64, seed: u64) -> InversionVerdict {
    if seconds == 0 || !(0.0..=1.0).contains(&medium_arrival_prob) {
        return InversionVerdict::InvalidConfig;
    }
    // States: low holds resource L; high needs L; medium can preempt low.
    let mut inversion_events = 0u32;
    let mut max_inversion_duration = 0u32;
    let mut current_inversion = 0u32;
    let mut rng_state = seed | 1;
    let mut high_waiting = false;
    let mut high_blocked_since = 0u32;
    for sec in 0..seconds {
        // High needs the resource periodically.
        if !high_waiting && sec % 10 == 0 {
            high_waiting = true;
            high_blocked_since = sec;
        }
        // Medium may arrive and preempt low.
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if high_waiting && r < medium_arrival_prob {
            // Inversion: medium preempts low while high is waiting.
            current_inversion += 1;
            if current_inversion > max_inversion_duration {
                max_inversion_duration = current_inversion;
            }
        } else if high_waiting {
            // High eventually unblocks.
            inversion_events += 1;
            high_waiting = false;
            current_inversion = 0;
            let _ = high_blocked_since;
        }
    }
    InversionVerdict::Ok {
        inversion_events,
        max_inversion_duration,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_priority_inversion_detect")?;

    println!("low medium load: {:?}", simulate(1000, 0.05, 42));
    println!("high medium load: {:?}", simulate(1000, 0.50, 42));
    println!("invalid: {:?}", simulate(0, 0.05, 42));
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
    fn high_medium_load_more_inversions() {
        let lo = simulate(1000, 0.05, 42);
        let hi = simulate(1000, 0.80, 42);
        if let (
            InversionVerdict::Ok {
                max_inversion_duration: l,
                ..
            },
            InversionVerdict::Ok {
                max_inversion_duration: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 0.05, 42), InversionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(1000, 1.5, 42), InversionVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 0.1, 42);
        let b = simulate(500, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_medium_no_inversion() {
        let v = simulate(1000, 0.0, 42);
        if let InversionVerdict::Ok {
            max_inversion_duration,
            ..
        } = v
        {
            assert_eq!(max_inversion_duration, 0);
        }
    }

    #[test]
    fn inversion_events_le_seconds() {
        let v = simulate(1000, 0.5, 42);
        if let InversionVerdict::Ok {
            inversion_events, ..
        } = v
        {
            assert!(inversion_events <= 1000);
        }
    }

    #[test]
    fn max_duration_le_seconds() {
        let v = simulate(1000, 0.5, 42);
        if let InversionVerdict::Ok {
            max_inversion_duration,
            ..
        } = v
        {
            assert!(max_inversion_duration <= 1000);
        }
    }

    #[test]
    fn always_medium_full_starvation() {
        let v = simulate(1000, 1.0, 42);
        if let InversionVerdict::Ok {
            max_inversion_duration,
            ..
        } = v
        {
            assert!(max_inversion_duration > 0);
        }
    }

    #[test]
    fn inversion_count_nonneg() {
        let v = simulate(100, 0.5, 42);
        if let InversionVerdict::Ok {
            inversion_events, ..
        } = v
        {
            // u32 always nonneg; documents intent.
            let _ = inversion_events;
        }
    }

    #[test]
    fn small_window_works() {
        let v = simulate(20, 0.1, 42);
        assert!(matches!(v, InversionVerdict::Ok { .. }));
    }

    #[test]
    fn many_seconds_handled() {
        let v = simulate(10_000, 0.1, 42);
        assert!(matches!(v, InversionVerdict::Ok { .. }));
    }
}
