//! # Data Streaming Buffer Size Picker
//!
//! Streaming dataloader uses a ring buffer to overlap I/O with compute.
//! Buffer size picks: too small = compute stalls on I/O; too large =
//! memory waste + stale data. Heuristic: buffer = max(4 × batch_size,
//! ceil(IO_latency_ms / step_latency_ms × batch_size)). This recipe
//! builds the picker.
//!
//! Demonstrates the **DATA.19** recipe for PMAT-132 (data-loading coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DATA-001 + Little's Law applied to streaming.
//!
//! Run with: cargo run --example data_streaming_buffer_size
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_MULTIPLIER: u32 = 4;
const HARD_CAP: u32 = 1024;

#[derive(Debug, PartialEq)]
pub enum BufferVerdict {
    Ok { samples: u32 },
    InvalidBatch,
    InvalidLatency,
    AboveHardCap { recommended: u32 },
}

pub fn pick(batch_size: u32, io_latency_ms: f64, step_latency_ms: f64) -> BufferVerdict {
    if batch_size == 0 {
        return BufferVerdict::InvalidBatch;
    }
    if !io_latency_ms.is_finite() || !step_latency_ms.is_finite() {
        return BufferVerdict::InvalidLatency;
    }
    if io_latency_ms < 0.0 || step_latency_ms <= 0.0 {
        return BufferVerdict::InvalidLatency;
    }
    let by_latency = (io_latency_ms / step_latency_ms * f64::from(batch_size)).ceil() as u32;
    let buffer = (MIN_MULTIPLIER * batch_size).max(by_latency);
    if buffer > HARD_CAP {
        return BufferVerdict::AboveHardCap {
            recommended: HARD_CAP,
        };
    }
    BufferVerdict::Ok { samples: buffer }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("data_streaming_buffer_size")?;

    let cases = [
        (32u32, 100.0_f64, 50.0_f64),
        (32, 500.0, 50.0),
        (32, 5000.0, 50.0),
        (0, 100.0, 50.0),
        (32, 100.0, 0.0),
    ];
    for (b, io, step) in cases {
        println!(
            "batch={b} io={io}ms step={step}ms  →  {:?}",
            pick(b, io, step)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fast_io_uses_min_multiplier() {
        // io < step → 4× min wins.
        let v = pick(32, 10.0, 50.0);
        assert!(matches!(v, BufferVerdict::Ok { samples: 128 }));
    }

    #[test]
    fn slow_io_grows_buffer() {
        // io = 500ms, step = 50ms, batch = 32 → 500/50 × 32 = 320 wins over 4×32 = 128.
        let v = pick(32, 500.0, 50.0);
        assert!(matches!(v, BufferVerdict::Ok { samples: 320 }));
    }

    #[test]
    fn excessive_io_capped() {
        // io = 5000ms, step = 1ms, batch = 32 → 160_000, capped at HARD_CAP.
        let v = pick(32, 5000.0, 1.0);
        assert!(matches!(
            v,
            BufferVerdict::AboveHardCap {
                recommended: HARD_CAP
            }
        ));
    }

    #[test]
    fn zero_batch_invalid() {
        assert_eq!(pick(0, 100.0, 50.0), BufferVerdict::InvalidBatch);
    }

    #[test]
    fn zero_step_invalid() {
        assert_eq!(pick(32, 100.0, 0.0), BufferVerdict::InvalidLatency);
    }

    #[test]
    fn negative_io_invalid() {
        assert_eq!(pick(32, -1.0, 50.0), BufferVerdict::InvalidLatency);
    }

    #[test]
    fn nan_latency_invalid() {
        assert_eq!(pick(32, f64::NAN, 50.0), BufferVerdict::InvalidLatency);
    }

    #[test]
    fn zero_io_uses_min_multiplier() {
        let v = pick(32, 0.0, 50.0);
        assert!(matches!(v, BufferVerdict::Ok { samples: 128 }));
    }

    #[test]
    fn small_batch_handled() {
        let v = pick(1, 10.0, 5.0);
        assert!(matches!(v, BufferVerdict::Ok { .. }));
    }

    #[test]
    fn at_hard_cap_passes() {
        // Exactly HARD_CAP allowed.
        let v = pick(256, 200.0, 50.0);
        // 4 × 256 = 1024 = HARD_CAP.
        assert!(matches!(v, BufferVerdict::Ok { samples: 1024 }));
    }
}
