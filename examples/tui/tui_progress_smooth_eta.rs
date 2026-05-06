//! # TUI Progress Smoothed ETA
//!
//! Smooth ETA using exponential moving average over recent rate
//! samples. Stabilizes the displayed ETA against jitter.
//!
//! Demonstrates the **TUI.40** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: indicatif --eta exponential smoothing.
//!
//! Run with: cargo run --example tui_progress_smooth_eta
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmoothEtaVerdict {
    Ok { smoothed_rate: f64, eta_secs: f64 },
    Complete,
    InvalidConfig,
}

pub fn update(
    previous_rate: f64,
    new_rate: f64,
    alpha: f64,
    completed: u64,
    total: u64,
) -> SmoothEtaVerdict {
    if !alpha.is_finite()
        || !(0.0..=1.0).contains(&alpha)
        || !previous_rate.is_finite()
        || !new_rate.is_finite()
        || previous_rate < 0.0
        || new_rate < 0.0
        || total == 0
    {
        return SmoothEtaVerdict::InvalidConfig;
    }
    if completed >= total {
        return SmoothEtaVerdict::Complete;
    }
    let smoothed_rate = alpha * new_rate + (1.0 - alpha) * previous_rate;
    let remaining = (total - completed) as f64;
    let eta_secs = if smoothed_rate > 0.0 {
        remaining / smoothed_rate
    } else {
        f64::INFINITY
    };
    SmoothEtaVerdict::Ok {
        smoothed_rate,
        eta_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_smooth_eta")?;

    println!("smooth: {:?}", update(10.0, 12.0, 0.3, 50, 100));
    println!("complete: {:?}", update(10.0, 12.0, 0.3, 100, 100));
    println!("invalid: {:?}", update(10.0, 12.0, 1.5, 50, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updater_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ema_blends_rates() {
        // alpha = 0.5 → smoothed = 0.5*12 + 0.5*10 = 11.
        let v = update(10.0, 12.0, 0.5, 0, 100);
        if let SmoothEtaVerdict::Ok { smoothed_rate, .. } = v {
            assert!((smoothed_rate - 11.0).abs() < 1e-9);
        }
    }

    #[test]
    fn alpha_zero_keeps_previous() {
        let v = update(10.0, 100.0, 0.0, 0, 100);
        if let SmoothEtaVerdict::Ok { smoothed_rate, .. } = v {
            assert!((smoothed_rate - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn alpha_one_uses_new() {
        let v = update(10.0, 100.0, 1.0, 0, 100);
        if let SmoothEtaVerdict::Ok { smoothed_rate, .. } = v {
            assert!((smoothed_rate - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn complete_when_done() {
        assert_eq!(
            update(10.0, 10.0, 0.5, 100, 100),
            SmoothEtaVerdict::Complete
        );
    }

    #[test]
    fn invalid_alpha_over_one() {
        assert_eq!(
            update(10.0, 10.0, 1.5, 0, 100),
            SmoothEtaVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(
            update(10.0, 10.0, 0.5, 0, 0),
            SmoothEtaVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            update(f64::NAN, 10.0, 0.5, 0, 100),
            SmoothEtaVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_rate_infinite_eta() {
        let v = update(0.0, 0.0, 0.5, 0, 100);
        if let SmoothEtaVerdict::Ok { eta_secs, .. } = v {
            assert!(eta_secs.is_infinite());
        }
    }

    #[test]
    fn higher_rate_lower_eta() {
        let slow = update(0.0, 1.0, 1.0, 0, 100);
        let fast = update(0.0, 10.0, 1.0, 0, 100);
        if let (
            SmoothEtaVerdict::Ok { eta_secs: s, .. },
            SmoothEtaVerdict::Ok { eta_secs: f, .. },
        ) = (slow, fast)
        {
            assert!(f < s);
        }
    }

    #[test]
    fn deterministic() {
        let a = update(10.0, 12.0, 0.3, 50, 100);
        let b = update(10.0, 12.0, 0.3, 50, 100);
        assert_eq!(a, b);
    }
}
