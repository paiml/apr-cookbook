//! # Monte-Carlo Jitter Buffer Underrun
//!
//! Sim a jitter buffer that smooths variable arrival times. Returns
//! underrun count (buffer-empty events) and mean buffer occupancy.
//!
//! Demonstrates the **MC.174** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VoIP RFC 3550 RTP jitter; WebRTC NetEQ adaptive jitter
//!  buffer.
//!
//! Run with: cargo run --example mc_jitter_buffer_underrun
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum JitterVerdict {
    Ok {
        underruns: u32,
        mean_occupancy_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    target_size: u32,
    jitter_pct: u32,
    consume_rate_x100: u32,
    duration_ticks: u32,
    seed: u64,
) -> JitterVerdict {
    if target_size < 2 || jitter_pct >= 100 || consume_rate_x100 == 0 || duration_ticks < 100 {
        return JitterVerdict::InvalidConfig;
    }
    let consume = consume_rate_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut buffer = target_size as f64;
    let mut underruns = 0u32;
    let mut occupancy_sum = 0.0f64;
    for _ in 0..duration_ticks {
        // Arrival: target_size's worth of producer/consumer with jitter
        let jitter_u = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let jitter = (jitter_u - 0.5) * 2.0 * (jitter_pct as f64 / 100.0);
        let arrival = consume * (1.0 + jitter);
        buffer += arrival;
        buffer -= consume;
        if buffer < 0.0 {
            underruns += 1;
            buffer = 0.0;
        }
        if buffer > target_size as f64 * 2.0 {
            buffer = target_size as f64 * 2.0; // cap
        }
        occupancy_sum += buffer;
    }
    let mean = occupancy_sum / duration_ticks as f64 * 100.0;
    JitterVerdict::Ok {
        underruns,
        mean_occupancy_x100: mean as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_jitter_buffer_underrun")?;

    println!("low jitter: {:?}", simulate(10, 10, 100, 1000, 42));
    println!("high jitter: {:?}", simulate(10, 80, 100, 1000, 42));
    println!("invalid: {:?}", simulate(1, 10, 100, 1000, 42));
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
    fn invalid_too_small_buffer() {
        assert_eq!(simulate(1, 10, 100, 1000, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_jitter_at_100() {
        assert_eq!(
            simulate(10, 100, 100, 1000, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_consume() {
        assert_eq!(simulate(10, 10, 0, 1000, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_short_duration() {
        assert_eq!(simulate(10, 10, 100, 50, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10, 10, 100, 1000, 42);
        let b = simulate(10, 10, 100, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_jitter_more_underruns() {
        let low = simulate(10, 10, 100, 5000, 42);
        let high = simulate(10, 80, 100, 5000, 42);
        if let (JitterVerdict::Ok { underruns: l, .. }, JitterVerdict::Ok { underruns: h, .. }) =
            (low, high)
        {
            assert!(h >= l);
        }
    }

    #[test]
    fn no_jitter_no_underruns() {
        let v = simulate(10, 0, 100, 1000, 42);
        if let JitterVerdict::Ok { underruns, .. } = v {
            assert_eq!(underruns, 0);
        }
    }

    #[test]
    fn occupancy_finite() {
        let v = simulate(10, 10, 100, 1000, 42);
        if let JitterVerdict::Ok {
            mean_occupancy_x100,
            ..
        } = v
        {
            assert!(mean_occupancy_x100 < u32::MAX);
        }
    }

    #[test]
    fn underruns_le_duration() {
        let v = simulate(10, 80, 100, 1000, 42);
        if let JitterVerdict::Ok { underruns, .. } = v {
            assert!(underruns <= 1000);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(2, 0, 1, 100, 42);
        assert!(matches!(v, JitterVerdict::Ok { .. }));
    }

    #[test]
    fn many_ticks_handled() {
        let v = simulate(10, 10, 100, 100_000, 42);
        assert!(matches!(v, JitterVerdict::Ok { .. }));
    }
}
