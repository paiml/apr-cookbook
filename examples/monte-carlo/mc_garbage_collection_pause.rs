//! # Monte-Carlo GC Stop-the-World Pause
//!
//! Sim a runtime with periodic GC pauses: every `gc_interval_ms`,
//! pause for `pause_ms`. Reports total pause time and pause-rate
//! over the simulated window.
//!
//! Demonstrates the **MC.76** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HotSpot GC tuning guide; Bacon et al., "A Real-time
//!  Garbage Collector with Low Overhead" (POPL 2003).
//!
//! Run with: cargo run --example mc_garbage_collection_pause
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GcVerdict {
    Ok {
        pauses: u32,
        total_pause_ms: u32,
        pause_fraction: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    window_ms: u32,
    gc_interval_ms: u32,
    avg_pause_ms: u32,
    jitter_pct: u32,
    seed: u64,
) -> GcVerdict {
    if window_ms == 0 || gc_interval_ms == 0 || avg_pause_ms == 0 || jitter_pct > 100 {
        return GcVerdict::InvalidConfig;
    }
    let mut now: u32 = 0;
    let mut pauses: u32 = 0;
    let mut total_pause: u32 = 0;
    let mut next_gc = gc_interval_ms;
    let mut rng_state = seed | 1;
    while now < window_ms {
        if now >= next_gc {
            let drift = (lcg(&mut rng_state) >> 32) as u32 % 100;
            let span = (avg_pause_ms * jitter_pct) / 100;
            let lo = avg_pause_ms.saturating_sub(span);
            let pause = lo + (drift * span * 2 / 100).min(2 * span);
            total_pause += pause;
            pauses += 1;
            now += pause;
            next_gc = now + gc_interval_ms;
        } else {
            now = (now + 1).min(window_ms);
            if now == window_ms {
                break;
            }
            // Skip forward to the next gc time to keep loop bounded.
            now = next_gc.min(window_ms);
        }
    }
    let pause_fraction = f64::from(total_pause) / f64::from(window_ms);
    GcVerdict::Ok {
        pauses,
        total_pause_ms: total_pause,
        pause_fraction,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_garbage_collection_pause")?;

    println!("low pause: {:?}", simulate(60_000, 1000, 10, 20, 42));
    println!("high pause: {:?}", simulate(60_000, 100, 50, 20, 42));
    println!("invalid: {:?}", simulate(0, 1000, 10, 20, 42));
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
    fn longer_window_more_pauses() {
        let short = simulate(10_000, 1000, 10, 0, 42);
        let long = simulate(100_000, 1000, 10, 0, 42);
        if let (GcVerdict::Ok { pauses: s, .. }, GcVerdict::Ok { pauses: l, .. }) = (short, long) {
            assert!(l > s);
        }
    }

    #[test]
    fn longer_pause_more_total_time() {
        let lo = simulate(60_000, 1000, 10, 0, 42);
        let hi = simulate(60_000, 1000, 100, 0, 42);
        if let (
            GcVerdict::Ok {
                total_pause_ms: l, ..
            },
            GcVerdict::Ok {
                total_pause_ms: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(simulate(0, 1000, 10, 20, 42), GcVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(simulate(60_000, 0, 10, 20, 42), GcVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_pause() {
        assert_eq!(simulate(60_000, 1000, 0, 20, 42), GcVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_jitter_above_100() {
        assert_eq!(
            simulate(60_000, 1000, 10, 200, 42),
            GcVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(60_000, 1000, 10, 20, 42);
        let b = simulate(60_000, 1000, 10, 20, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn pause_fraction_in_unit_range() {
        let v = simulate(60_000, 1000, 10, 20, 42);
        if let GcVerdict::Ok { pause_fraction, .. } = v {
            assert!((0.0..=1.0).contains(&pause_fraction));
        }
    }

    #[test]
    fn total_pause_le_window() {
        let v = simulate(60_000, 100, 50, 0, 42);
        if let GcVerdict::Ok { total_pause_ms, .. } = v {
            assert!(total_pause_ms <= 60_000);
        }
    }

    #[test]
    fn very_long_interval_few_pauses() {
        let v = simulate(60_000, 100_000, 10, 0, 42);
        if let GcVerdict::Ok { pauses, .. } = v {
            assert!(pauses <= 1);
        }
    }

    #[test]
    fn zero_jitter_constant_pause() {
        let a = simulate(60_000, 1000, 10, 0, 42);
        let b = simulate(60_000, 1000, 10, 0, 99);
        // With zero jitter, two seeds produce same total pause.
        if let (
            GcVerdict::Ok {
                total_pause_ms: x, ..
            },
            GcVerdict::Ok {
                total_pause_ms: y, ..
            },
        ) = (a, b)
        {
            assert_eq!(x, y);
        }
    }
}
