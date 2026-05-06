//! # TUI Progress Bar State
//!
//! Compute progress-bar state from completed/total work units.
//! Returns ratio, formatted percent, and ETA seconds (linear extrapolation
//! from elapsed). Pure function — no terminal IO.
//!
//! Demonstrates the **TUI.01** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: indicatif progress-bar conventions.
//!
//! Run with: cargo run --example tui_progress_state
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ProgressVerdict {
    Ok {
        ratio: f64,
        percent: u32,
        eta_secs: f64,
    },
    Complete,
    InvalidInput,
}

pub fn compute(completed: u64, total: u64, elapsed_secs: f64) -> ProgressVerdict {
    if total == 0 || !elapsed_secs.is_finite() || elapsed_secs < 0.0 {
        return ProgressVerdict::InvalidInput;
    }
    if completed >= total {
        return ProgressVerdict::Complete;
    }
    let ratio = completed as f64 / total as f64;
    let percent = (ratio * 100.0) as u32;
    let eta_secs = if completed == 0 {
        f64::INFINITY
    } else {
        elapsed_secs * (total - completed) as f64 / completed as f64
    };
    ProgressVerdict::Ok {
        ratio,
        percent,
        eta_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_state")?;

    println!("at start: {:?}", compute(0, 100, 1.0));
    println!("halfway: {:?}", compute(50, 100, 10.0));
    println!("near end: {:?}", compute(99, 100, 99.0));
    println!("complete: {:?}", compute(100, 100, 100.0));
    println!("invalid: {:?}", compute(50, 0, 10.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn halfway_50_pct() {
        let v = compute(50, 100, 10.0);
        if let ProgressVerdict::Ok {
            ratio,
            percent,
            eta_secs,
        } = v
        {
            assert!((ratio - 0.5).abs() < 1e-9);
            assert_eq!(percent, 50);
            // 10 secs to do 50; remaining 50 → 10 secs.
            assert!((eta_secs - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn complete_returned_when_done() {
        assert_eq!(compute(100, 100, 100.0), ProgressVerdict::Complete);
    }

    #[test]
    fn over_total_complete() {
        assert_eq!(compute(150, 100, 100.0), ProgressVerdict::Complete);
    }

    #[test]
    fn zero_total_invalid() {
        assert_eq!(compute(50, 0, 10.0), ProgressVerdict::InvalidInput);
    }

    #[test]
    fn negative_elapsed_invalid() {
        assert_eq!(compute(50, 100, -1.0), ProgressVerdict::InvalidInput);
    }

    #[test]
    fn nan_elapsed_invalid() {
        assert_eq!(compute(50, 100, f64::NAN), ProgressVerdict::InvalidInput);
    }

    #[test]
    fn zero_completed_infinite_eta() {
        let v = compute(0, 100, 1.0);
        if let ProgressVerdict::Ok { eta_secs, .. } = v {
            assert!(eta_secs.is_infinite());
        }
    }

    #[test]
    fn quarter_25_pct() {
        let v = compute(25, 100, 5.0);
        if let ProgressVerdict::Ok {
            percent, eta_secs, ..
        } = v
        {
            assert_eq!(percent, 25);
            // 5 secs for 25 → remaining 75 takes 15.
            assert!((eta_secs - 15.0).abs() < 1e-9);
        }
    }

    #[test]
    fn eta_decreases_as_progress_increases() {
        let early = compute(10, 100, 1.0);
        let late = compute(90, 100, 9.0);
        if let (
            ProgressVerdict::Ok { eta_secs: e1, .. },
            ProgressVerdict::Ok { eta_secs: e2, .. },
        ) = (early, late)
        {
            assert!(e1 > e2);
        }
    }

    #[test]
    fn zero_elapsed_works() {
        let v = compute(50, 100, 0.0);
        if let ProgressVerdict::Ok { eta_secs, .. } = v {
            // 0 elapsed for 50 done → eta = 0.
            assert!((eta_secs - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = compute(50, 100, 10.0);
        let b = compute(50, 100, 10.0);
        assert_eq!(a, b);
    }
}
