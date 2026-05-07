//! # TUI Progress ETA Remaining
//!
//! Compute estimated remaining time from elapsed seconds + percent
//! complete. Returns ETA secs and human-readable display.
//!
//! Demonstrates the **TUI.122** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pip/cargo progress bar ETA computation; tqdm Python
//!  progress bar.
//!
//! Run with: cargo run --example tui_progress_estimate_remaining
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EtaVerdict {
    Ok { eta_secs: u32, display: String },
    InvalidConfig,
}

pub fn estimate(elapsed_secs: u32, percent_complete: u32) -> EtaVerdict {
    if percent_complete == 0 || percent_complete > 100 {
        return EtaVerdict::InvalidConfig;
    }
    let total_secs = elapsed_secs * 100 / percent_complete;
    let eta_secs = total_secs.saturating_sub(elapsed_secs);
    let display = humanize(eta_secs);
    EtaVerdict::Ok { eta_secs, display }
}

fn humanize(secs: u32) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_estimate_remaining")?;

    println!("50%, 60s elapsed: {:?}", estimate(60, 50));
    println!("90%, 540s elapsed: {:?}", estimate(540, 90));
    println!("invalid: {:?}", estimate(60, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn eta_correct() {
        let v = estimate(60, 50);
        if let EtaVerdict::Ok { eta_secs, .. } = v {
            // 50% in 60s → total = 120s → ETA = 60s.
            assert_eq!(eta_secs, 60);
        }
    }

    #[test]
    fn nearly_done_low_eta() {
        let v = estimate(540, 90);
        if let EtaVerdict::Ok { eta_secs, .. } = v {
            // 90% in 540s → total = 600s → ETA = 60s.
            assert_eq!(eta_secs, 60);
        }
    }

    #[test]
    fn invalid_zero_percent() {
        assert_eq!(estimate(60, 0), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_above_100_percent() {
        assert_eq!(estimate(60, 150), EtaVerdict::InvalidConfig);
    }

    #[test]
    fn at_100_percent_zero_eta() {
        let v = estimate(60, 100);
        if let EtaVerdict::Ok { eta_secs, .. } = v {
            assert_eq!(eta_secs, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = estimate(60, 50);
        let r2 = estimate(60, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn display_uses_minutes_when_under_hour() {
        let v = estimate(60, 50);
        if let EtaVerdict::Ok { display, .. } = v {
            assert!(display.contains('m'));
        }
    }

    #[test]
    fn display_uses_hours_when_over_hour() {
        let v = estimate(3600, 25);
        if let EtaVerdict::Ok { display, .. } = v {
            // ETA = 3600 * (75/25) = 10800s = 3h.
            assert!(display.contains('h'));
        }
    }

    #[test]
    fn display_uses_seconds_when_under_minute() {
        let v = estimate(60, 99);
        if let EtaVerdict::Ok { display, .. } = v {
            assert!(display.ends_with('s'));
        }
    }

    #[test]
    fn zero_elapsed_zero_eta() {
        let v = estimate(0, 50);
        if let EtaVerdict::Ok { eta_secs, .. } = v {
            assert_eq!(eta_secs, 0);
        }
    }

    #[test]
    fn higher_percent_lower_eta() {
        let lo = estimate(60, 25);
        let hi = estimate(60, 75);
        if let (EtaVerdict::Ok { eta_secs: l, .. }, EtaVerdict::Ok { eta_secs: h, .. }) = (lo, hi) {
            assert!(h < l);
        }
    }
}
