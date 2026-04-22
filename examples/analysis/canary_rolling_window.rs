//! # Recipe: Rolling-Window Canary Checks
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr canary watch model.apr --window 50 --alert-threshold 3`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example canary_rolling_window` exits 0
//! 2. [x] `cargo test --example canary_rolling_window` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr canary` behavior in-process (no shell-out)
//! 10. [x] Unit tests cover window slide, alarm trigger, clean window
//!
//! ## Learning Objective
//! Simulates continuous production canary monitoring: runs N checks, slides a
//! fixed-size window over the failure timeline, and fires an alarm the first
//! time consecutive failures exceed an alert threshold. This models "health
//! checks over time" rather than a one-shot pass/fail.
//!
//! ## Run Command
//! ```bash
//! cargo run --example canary_rolling_window
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::Rng;
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Tick {
    t: usize,
    passed: bool,
    score: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowState {
    window_size: usize,
    alert_threshold: usize,
    fails_in_window: usize,
    alarm_at: Option<usize>,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

/// Produce a deterministic sequence of canary check outcomes.
/// The sequence has a "drift period" in the middle where failures cluster.
fn generate_timeline(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<Tick> {
    (0..n)
        .map(|t| {
            let score: f32 = rng.gen_range(0.0..1.0);
            // Fail-rate increases during 40..60% of the run to simulate a drift period.
            let drift_mid = n / 2;
            let in_drift_zone = t >= drift_mid.saturating_sub(n / 10) && t <= drift_mid + n / 10;
            let fail_prob = if in_drift_zone { 0.55 } else { 0.05 };
            let passed = score > fail_prob as f32;
            Tick { t, passed, score }
        })
        .collect()
}

/// Slide a fixed-size window across the timeline, firing an alarm the first time
/// the count of failures inside the window reaches the threshold.
fn scan_window(timeline: &[Tick], window_size: usize, alert_threshold: usize) -> WindowState {
    let mut fails_in_window = 0_usize;
    let mut alarm_at: Option<usize> = None;
    let mut buf: std::collections::VecDeque<bool> =
        std::collections::VecDeque::with_capacity(window_size);

    for tick in timeline {
        let is_fail = !tick.passed;
        if is_fail {
            fails_in_window += 1;
        }
        buf.push_back(is_fail);
        if buf.len() > window_size {
            if let Some(old) = buf.pop_front() {
                if old {
                    fails_in_window = fails_in_window.saturating_sub(1);
                }
            }
        }
        if alarm_at.is_none() && fails_in_window >= alert_threshold {
            alarm_at = Some(tick.t);
        }
    }

    WindowState {
        window_size,
        alert_threshold,
        fails_in_window,
        alarm_at,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("canary_rolling_window")?;
    println!("=== Recipe: {} ===", ctx.name());

    let n_ticks = 120_usize;
    let window_size = 20_usize;
    let alert_threshold = 5_usize;

    let timeline = generate_timeline(ctx.rng(), n_ticks);
    let passed = timeline.iter().filter(|t| t.passed).count();
    let failed = timeline.len() - passed;

    println!("Ticks: {n_ticks}, passed: {passed}, failed: {failed}");
    println!(
        "Window: {}, alert_threshold: {}",
        window_size, alert_threshold
    );

    let state = scan_window(&timeline, window_size, alert_threshold);
    match state.alarm_at {
        Some(t) => println!("ALARM fired at tick t={t}"),
        None => println!("No alarm — canary remained healthy."),
    }

    // Sanity: with intentional drift, an alarm should fire.
    assert!(
        state.alarm_at.is_some(),
        "rolling window should alert on injected drift"
    );

    let out = json!({
        "recipe": ctx.name(),
        "n_ticks": n_ticks,
        "window_size": window_size,
        "alert_threshold": alert_threshold,
        "passed": passed,
        "failed": failed,
        "alarm_at": state.alarm_at,
        "sample_ticks": timeline.iter().take(8).map(|t| json!({
            "t": t.t,
            "passed": t.passed,
            "score": t.score,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("rolling.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.record_metric("n_ticks", n_ticks as i64);
    ctx.record_metric("window_size", window_size as i64);
    ctx.record_metric("alert_threshold", alert_threshold as i64);
    ctx.record_metric("alarm_at", state.alarm_at.map_or(-1, |t| t as i64));

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_clean_timeline_no_alarm() {
        let timeline: Vec<Tick> = (0..30)
            .map(|t| Tick {
                t,
                passed: true,
                score: 1.0,
            })
            .collect();
        let state = scan_window(&timeline, 10, 3);
        assert!(state.alarm_at.is_none());
    }

    #[test]
    fn test_all_failures_trigger_alarm_early() {
        let timeline: Vec<Tick> = (0..10)
            .map(|t| Tick {
                t,
                passed: false,
                score: 0.0,
            })
            .collect();
        let state = scan_window(&timeline, 5, 3);
        assert_eq!(state.alarm_at, Some(2)); // 3 fails cumulative.
    }

    #[test]
    fn test_window_slides_and_forgets_old_fails() {
        // 3 fails then a long clean run: alarm should still fire at t=2.
        let mut ticks: Vec<Tick> = (0..3)
            .map(|t| Tick {
                t,
                passed: false,
                score: 0.0,
            })
            .collect();
        for t in 3..50 {
            ticks.push(Tick {
                t,
                passed: true,
                score: 1.0,
            });
        }
        let state = scan_window(&ticks, 5, 3);
        assert_eq!(state.alarm_at, Some(2));
    }

    #[test]
    fn test_window_forgets_single_old_fail() {
        // Single fail then window slides past it: no alarm at threshold 2.
        let mut ticks: Vec<Tick> = vec![Tick {
            t: 0,
            passed: false,
            score: 0.0,
        }];
        for t in 1..20 {
            ticks.push(Tick {
                t,
                passed: true,
                score: 1.0,
            });
        }
        let state = scan_window(&ticks, 3, 2);
        assert!(state.alarm_at.is_none());
    }

    #[test]
    fn test_generate_timeline_length() {
        let mut rng = make_rng();
        let tl = generate_timeline(&mut rng, 50);
        assert_eq!(tl.len(), 50);
    }

    #[test]
    fn test_generate_timeline_deterministic() {
        let a = generate_timeline(&mut make_rng(), 40);
        let b = generate_timeline(&mut make_rng(), 40);
        let a_pass: Vec<bool> = a.iter().map(|t| t.passed).collect();
        let b_pass: Vec<bool> = b.iter().map(|t| t.passed).collect();
        assert_eq!(a_pass, b_pass);
    }
}
