//! # GPU CUDA Graph Capture Decision
//!
//! CUDA graphs amortize launch overhead by capturing once + replaying.
//! Decision rule:
//!   capture_overhead_us / replay_count < launch_savings_us → Capture
//!   otherwise → Skip (per-launch better)
//!
//! Demonstrates the **GPU.38** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CUDA C Programming Guide § Graph capture.
//!
//! Run with: cargo run --example gpu_cuda_graph_capture
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CaptureVerdict {
    Capture {
        savings_us_per_replay: f64,
        breakeven_at_replays: u32,
    },
    SkipCapture {
        reason: &'static str,
    },
    InvalidInput,
}

pub fn decide(capture_overhead_us: f64, per_launch_us: f64, replay_count: u32) -> CaptureVerdict {
    if !capture_overhead_us.is_finite()
        || !per_launch_us.is_finite()
        || capture_overhead_us < 0.0
        || per_launch_us < 0.0
    {
        return CaptureVerdict::InvalidInput;
    }
    if per_launch_us == 0.0 {
        return CaptureVerdict::SkipCapture {
            reason: "per-launch already zero",
        };
    }
    if replay_count == 0 {
        return CaptureVerdict::InvalidInput;
    }
    let breakeven_at_replays = (capture_overhead_us / per_launch_us).ceil() as u32;
    if replay_count >= breakeven_at_replays {
        let savings_us_per_replay = per_launch_us;
        CaptureVerdict::Capture {
            savings_us_per_replay,
            breakeven_at_replays,
        }
    } else {
        CaptureVerdict::SkipCapture {
            reason: "too few replays to amortize capture cost",
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_cuda_graph_capture")?;

    println!("many replays: {:?}", decide(1000.0, 10.0, 1000));
    println!("few replays: {:?}", decide(1000.0, 10.0, 50));
    println!("invalid: {:?}", decide(-1.0, 10.0, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decider_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn many_replays_capture() {
        let v = decide(1000.0, 10.0, 1000);
        assert!(matches!(v, CaptureVerdict::Capture { .. }));
    }

    #[test]
    fn few_replays_skip() {
        let v = decide(1000.0, 10.0, 50);
        assert!(matches!(v, CaptureVerdict::SkipCapture { .. }));
    }

    #[test]
    fn invalid_negative_overhead() {
        assert_eq!(decide(-1.0, 10.0, 1000), CaptureVerdict::InvalidInput);
    }

    #[test]
    fn invalid_negative_launch() {
        assert_eq!(decide(1000.0, -10.0, 1000), CaptureVerdict::InvalidInput);
    }

    #[test]
    fn invalid_zero_replays() {
        assert_eq!(decide(1000.0, 10.0, 0), CaptureVerdict::InvalidInput);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(decide(f64::NAN, 10.0, 100), CaptureVerdict::InvalidInput);
    }

    #[test]
    fn breakeven_correct() {
        // 1000 / 10 = 100 replays.
        let v = decide(1000.0, 10.0, 100);
        if let CaptureVerdict::Capture {
            breakeven_at_replays,
            ..
        } = v
        {
            assert_eq!(breakeven_at_replays, 100);
        }
    }

    #[test]
    fn at_breakeven_captures() {
        let v = decide(1000.0, 10.0, 100);
        assert!(matches!(v, CaptureVerdict::Capture { .. }));
    }

    #[test]
    fn just_below_breakeven_skips() {
        let v = decide(1000.0, 10.0, 99);
        assert!(matches!(v, CaptureVerdict::SkipCapture { .. }));
    }

    #[test]
    fn zero_per_launch_skip() {
        let v = decide(100.0, 0.0, 1000);
        assert!(matches!(v, CaptureVerdict::SkipCapture { .. }));
    }

    #[test]
    fn savings_equal_per_launch() {
        let v = decide(1000.0, 10.0, 1000);
        if let CaptureVerdict::Capture {
            savings_us_per_replay,
            ..
        } = v
        {
            assert!((savings_us_per_replay - 10.0).abs() < 1e-9);
        }
    }
}
