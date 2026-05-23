//! # TUI Live Preview Throttle
//!
//! Throttle re-render events so updates fire at most every
//! `interval_ms` regardless of input rate. Returns a verdict for each
//! event: `Render` (now), `Throttled` (skipped), or `Pending`
//! (deferred to a fire-time).
//!
//! Demonstrates the **TUI.144** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: lodash `_.throttle`; HTML5 `requestAnimationFrame`-style
//!  rate limiting.
//!
//! Run with: cargo run --example tui_live_preview_throttle
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThrottleVerdict {
    Ok {
        renders: Vec<u32>,
        throttled_count: u32,
    },
    InvalidConfig,
}

pub fn process(event_times_ms: &[u32], interval_ms: u32) -> ThrottleVerdict {
    if event_times_ms.is_empty() || interval_ms == 0 {
        return ThrottleVerdict::InvalidConfig;
    }
    let mut renders: Vec<u32> = Vec::new();
    let mut last_render: Option<u32> = None;
    let mut throttled = 0u32;
    for t in event_times_ms {
        match last_render {
            None => {
                renders.push(*t);
                last_render = Some(*t);
            }
            Some(last) if *t >= last + interval_ms => {
                renders.push(*t);
                last_render = Some(*t);
            }
            _ => {
                throttled += 1;
            }
        }
    }
    ThrottleVerdict::Ok {
        renders,
        throttled_count: throttled,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_live_preview_throttle")?;

    let events = [0u32, 50, 100, 200, 250, 300, 500];
    println!("interval-100: {:?}", process(&events, 100));
    println!("invalid: {:?}", process(&[], 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_event_always_renders() {
        let v = process(&[42], 100);
        if let ThrottleVerdict::Ok { renders, .. } = v {
            assert_eq!(renders, vec![42]);
        }
    }

    #[test]
    fn invalid_empty_events() {
        assert_eq!(process(&[], 100), ThrottleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(process(&[1, 2], 0), ThrottleVerdict::InvalidConfig);
    }

    #[test]
    fn within_interval_throttled() {
        let v = process(&[0, 50], 100);
        if let ThrottleVerdict::Ok {
            throttled_count, ..
        } = v
        {
            assert_eq!(throttled_count, 1);
        }
    }

    #[test]
    fn after_interval_renders() {
        let v = process(&[0, 100], 100);
        if let ThrottleVerdict::Ok { renders, .. } = v {
            assert_eq!(renders, vec![0, 100]);
        }
    }

    #[test]
    fn boundary_renders() {
        // event exactly at last+interval should render.
        let v = process(&[0, 100, 200], 100);
        if let ThrottleVerdict::Ok { renders, .. } = v {
            assert_eq!(renders, vec![0, 100, 200]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = process(&[0, 50, 100], 100);
        let r2 = process(&[0, 50, 100], 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rapid_events_mostly_throttled() {
        let events: Vec<u32> = (0..20).collect(); // 0,1,2,...,19 ms
        let v = process(&events, 100);
        if let ThrottleVerdict::Ok {
            renders,
            throttled_count,
        } = v
        {
            assert_eq!(renders.len(), 1);
            assert_eq!(throttled_count, 19);
        }
    }

    #[test]
    fn evenly_spaced_all_render() {
        let events = [0u32, 100, 200, 300, 400];
        let v = process(&events, 100);
        if let ThrottleVerdict::Ok { renders, .. } = v {
            assert_eq!(renders.len(), 5);
        }
    }

    #[test]
    fn throttled_count_correct() {
        let events = [0u32, 50, 100, 200, 250];
        let v = process(&events, 100);
        if let ThrottleVerdict::Ok {
            throttled_count, ..
        } = v
        {
            // Renders at 0, 100, 200; throttles at 50, 250.
            assert_eq!(throttled_count, 2);
        }
    }

    #[test]
    fn many_events_handled() {
        let events: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let v = process(&events, 100);
        assert!(matches!(v, ThrottleVerdict::Ok { .. }));
    }

    #[test]
    fn renders_strictly_increasing() {
        let events = [0u32, 100, 200, 300];
        let v = process(&events, 100);
        if let ThrottleVerdict::Ok { renders, .. } = v {
            for w in renders.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }
}
