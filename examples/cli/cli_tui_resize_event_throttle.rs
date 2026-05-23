//! # apr tui — Resize Event Throttle
//!
//! Terminal resize events fire continuously while dragging. Re-rendering
//! every event jacks CPU; throttling to ≥ 50 ms (~ 20 fps) is the
//! practical floor. Coalesce queued events: only the latest dimensions
//! matter. This recipe builds the throttle decision + coalescer.
//!
//! Demonstrates the **TUI.5** recipe for PMAT-122 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-001 + xterm SIGWINCH conventions
//!
//! Run with: cargo run --example cli_tui_resize_event_throttle
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_INTERVAL_MS: u64 = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeEvent {
    pub width: u16,
    pub height: u16,
    pub timestamp_ms: u64,
}

#[derive(Debug, PartialEq)]
pub enum ThrottleVerdict {
    Render,
    Throttle { wait_ms: u64 },
    InvalidDimensions,
}

pub fn decide(now_ms: u64, last_render_ms: Option<u64>, dims: (u16, u16)) -> ThrottleVerdict {
    if dims.0 == 0 || dims.1 == 0 {
        return ThrottleVerdict::InvalidDimensions;
    }
    match last_render_ms {
        None => ThrottleVerdict::Render,
        Some(t) if now_ms.saturating_sub(t) >= MIN_INTERVAL_MS => ThrottleVerdict::Render,
        Some(t) => ThrottleVerdict::Throttle {
            wait_ms: MIN_INTERVAL_MS - (now_ms - t),
        },
    }
}

pub fn coalesce(events: &[ResizeEvent]) -> Option<ResizeEvent> {
    events
        .iter()
        .filter(|e| e.width > 0 && e.height > 0)
        .max_by_key(|e| e.timestamp_ms)
        .copied()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_resize_event_throttle")?;

    println!("first event: {:?}", decide(100, None, (80, 24)));
    println!("75ms later: {:?}", decide(175, Some(100), (80, 24)));
    println!("20ms later: {:?}", decide(120, Some(100), (80, 24)));
    println!("zero-dim: {:?}", decide(100, None, (0, 24)));

    let events = [
        ResizeEvent {
            width: 80,
            height: 24,
            timestamp_ms: 100,
        },
        ResizeEvent {
            width: 90,
            height: 30,
            timestamp_ms: 150,
        },
        ResizeEvent {
            width: 100,
            height: 35,
            timestamp_ms: 200,
        },
    ];
    println!("coalesced: {:?}", coalesce(&events));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn first_event_renders_immediately() {
        assert_eq!(decide(100, None, (80, 24)), ThrottleVerdict::Render);
    }

    #[test]
    fn elapsed_below_floor_throttles() {
        let v = decide(120, Some(100), (80, 24));
        assert!(matches!(v, ThrottleVerdict::Throttle { wait_ms: 30 }));
    }

    #[test]
    fn elapsed_at_floor_renders() {
        assert_eq!(decide(150, Some(100), (80, 24)), ThrottleVerdict::Render);
    }

    #[test]
    fn elapsed_above_floor_renders() {
        assert_eq!(decide(500, Some(100), (80, 24)), ThrottleVerdict::Render);
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(
            decide(100, None, (0, 24)),
            ThrottleVerdict::InvalidDimensions
        );
    }

    #[test]
    fn zero_height_invalid() {
        assert_eq!(
            decide(100, None, (80, 0)),
            ThrottleVerdict::InvalidDimensions
        );
    }

    #[test]
    fn coalesce_picks_latest() {
        let events = [
            ResizeEvent {
                width: 80,
                height: 24,
                timestamp_ms: 100,
            },
            ResizeEvent {
                width: 90,
                height: 30,
                timestamp_ms: 150,
            },
            ResizeEvent {
                width: 100,
                height: 35,
                timestamp_ms: 200,
            },
        ];
        let c = coalesce(&events).unwrap();
        assert_eq!(c.timestamp_ms, 200);
        assert_eq!(c.width, 100);
    }

    #[test]
    fn coalesce_skips_invalid_events() {
        let events = [
            ResizeEvent {
                width: 0,
                height: 24,
                timestamp_ms: 200,
            },
            ResizeEvent {
                width: 80,
                height: 24,
                timestamp_ms: 100,
            },
        ];
        let c = coalesce(&events).unwrap();
        assert_eq!(c.width, 80);
    }

    #[test]
    fn coalesce_empty_returns_none() {
        assert!(coalesce(&[]).is_none());
    }

    #[test]
    fn coalesce_all_invalid_returns_none() {
        let events = [
            ResizeEvent {
                width: 0,
                height: 24,
                timestamp_ms: 100,
            },
            ResizeEvent {
                width: 80,
                height: 0,
                timestamp_ms: 200,
            },
        ];
        assert!(coalesce(&events).is_none());
    }
}
