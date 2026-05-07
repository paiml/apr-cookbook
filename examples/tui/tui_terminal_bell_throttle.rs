//! # TUI Terminal Bell Throttle
//!
//! Throttle terminal-bell events so they fire at most once per
//! `interval_ms`. Returns Bell-fire count and dropped count.
//!
//! Demonstrates the **TUI.170** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VT100 BEL (0x07) audible; tmux `bell-action` rate-limit
//!  semantics.
//!
//! Run with: cargo run --example tui_terminal_bell_throttle
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BellVerdict {
    Ok { fired: u32, dropped: u32 },
    InvalidConfig,
}

pub fn process(events_ms: &[u32], interval_ms: u32) -> BellVerdict {
    if events_ms.is_empty() || interval_ms == 0 {
        return BellVerdict::InvalidConfig;
    }
    let mut last_fire: Option<u32> = None;
    let mut fired = 0u32;
    let mut dropped = 0u32;
    for t in events_ms {
        match last_fire {
            None => {
                fired += 1;
                last_fire = Some(*t);
            }
            Some(prev) if *t >= prev + interval_ms => {
                fired += 1;
                last_fire = Some(*t);
            }
            _ => dropped += 1,
        }
    }
    BellVerdict::Ok { fired, dropped }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_terminal_bell_throttle")?;

    println!("clean: {:?}", process(&[0, 200, 400], 100));
    println!("throttled: {:?}", process(&[0, 50, 75, 200], 100));
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
    fn empty_input_rejected() {
        assert_eq!(process(&[], 100), BellVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(process(&[0], 0), BellVerdict::InvalidConfig);
    }

    #[test]
    fn first_event_fires() {
        let v = process(&[42], 100);
        if let BellVerdict::Ok { fired, dropped } = v {
            assert_eq!(fired, 1);
            assert_eq!(dropped, 0);
        }
    }

    #[test]
    fn within_interval_dropped() {
        let v = process(&[0, 50], 100);
        if let BellVerdict::Ok { dropped, .. } = v {
            assert_eq!(dropped, 1);
        }
    }

    #[test]
    fn after_interval_fires() {
        let v = process(&[0, 100], 100);
        if let BellVerdict::Ok { fired, .. } = v {
            assert_eq!(fired, 2);
        }
    }

    #[test]
    fn boundary_at_interval_fires() {
        let v = process(&[0, 100, 200], 100);
        if let BellVerdict::Ok { fired, .. } = v {
            assert_eq!(fired, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = process(&[0, 50], 100);
        let r2 = process(&[0, 50], 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rapid_burst_mostly_dropped() {
        let events: Vec<u32> = (0..20).collect();
        let v = process(&events, 100);
        if let BellVerdict::Ok { fired, dropped } = v {
            assert_eq!(fired, 1);
            assert_eq!(dropped, 19);
        }
    }

    #[test]
    fn evenly_spaced_all_fire() {
        let events = [0u32, 100, 200, 300, 400];
        let v = process(&events, 100);
        if let BellVerdict::Ok { fired, .. } = v {
            assert_eq!(fired, 5);
        }
    }

    #[test]
    fn many_events_handled() {
        let events: Vec<u32> = (0..1000).map(|i| i * 50).collect();
        let v = process(&events, 100);
        assert!(matches!(v, BellVerdict::Ok { .. }));
    }

    #[test]
    fn fired_plus_dropped_equals_events() {
        let v = process(&[0, 50, 100, 150, 200], 100);
        if let BellVerdict::Ok { fired, dropped } = v {
            assert_eq!(fired + dropped, 5);
        }
    }

    #[test]
    fn long_interval_more_dropped() {
        let events = [0u32, 100, 200];
        let short = process(&events, 50);
        let long = process(&events, 1000);
        if let (BellVerdict::Ok { fired: s, .. }, BellVerdict::Ok { fired: l, .. }) = (short, long)
        {
            assert!(l <= s);
        }
    }
}
