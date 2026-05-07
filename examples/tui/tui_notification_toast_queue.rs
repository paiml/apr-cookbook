//! # TUI Notification Toast Queue
//!
//! Process a stream of toast notifications with TTL, priority, and
//! a max-visible limit. Returns currently-visible toasts (top-priority
//! first) and a count dropped due to capacity.
//!
//! Demonstrates the **TUI.138** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Notification Center stacking; Android `Toast`
//!  queueing semantics; libnotify `NotifyNotification` priority.
//!
//! Run with: cargo run --example tui_notification_toast_queue
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ToastVerdict {
    Ok {
        visible: Vec<String>,
        dropped_count: u32,
    },
    InvalidConfig,
}

pub fn process(notifications: &[(&str, u8, u32)], now: u32, max_visible: u32) -> ToastVerdict {
    if max_visible == 0 {
        return ToastVerdict::InvalidConfig;
    }
    // Filter alive (expires_at > now), then sort by priority desc, message asc.
    let mut alive: Vec<(&&str, u8)> = notifications
        .iter()
        .filter(|(_, _, expires)| *expires > now)
        .map(|(msg, prio, _)| (msg, *prio))
        .collect();
    alive.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let total = alive.len() as u32;
    let dropped = total.saturating_sub(max_visible);
    let visible: Vec<String> = alive
        .into_iter()
        .take(max_visible as usize)
        .map(|(msg, _)| (*msg).to_string())
        .collect();
    ToastVerdict::Ok {
        visible,
        dropped_count: dropped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_notification_toast_queue")?;

    let n = [("error", 9u8, 100u32), ("info", 3, 100), ("warn", 5, 100)];
    println!("3 max: {:?}", process(&n, 50, 3));
    println!("1 max: {:?}", process(&n, 50, 1));
    println!("invalid: {:?}", process(&n, 50, 0));
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
    fn highest_priority_first() {
        let n = [("low", 1u8, 100u32), ("high", 9, 100)];
        let v = process(&n, 50, 2);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible[0], "high");
        }
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(process(&[], 0, 0), ToastVerdict::InvalidConfig);
    }

    #[test]
    fn expired_filtered() {
        let n = [("expired", 9u8, 10u32), ("alive", 1, 100)];
        let v = process(&n, 50, 5);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["alive".to_string()]);
        }
    }

    #[test]
    fn max_visible_truncates() {
        let n = [("a", 1u8, 100u32), ("b", 2, 100), ("c", 3, 100)];
        let v = process(&n, 50, 2);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible.len(), 2);
        }
    }

    #[test]
    fn dropped_count_correct() {
        let n = [
            ("a", 1u8, 100u32),
            ("b", 2, 100),
            ("c", 3, 100),
            ("d", 4, 100),
        ];
        let v = process(&n, 50, 2);
        if let ToastVerdict::Ok { dropped_count, .. } = v {
            assert_eq!(dropped_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let n = [("a", 1u8, 100u32)];
        let r1 = process(&n, 50, 5);
        let r2 = process(&n, 50, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_notifications_empty_visible() {
        let v = process(&[], 50, 5);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert!(visible.is_empty());
        }
    }

    #[test]
    fn equal_priority_alphabetical_tie_break() {
        let n = [("zeta", 5u8, 100u32), ("alpha", 5, 100)];
        let v = process(&n, 50, 2);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn boundary_expiry_filters() {
        // expires_at == now → expired (strictly greater required)
        let n = [("at_boundary", 1u8, 50u32)];
        let v = process(&n, 50, 5);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert!(visible.is_empty());
        }
    }

    #[test]
    fn dropped_zero_when_under_capacity() {
        let n = [("a", 1u8, 100u32)];
        let v = process(&n, 50, 5);
        if let ToastVerdict::Ok { dropped_count, .. } = v {
            assert_eq!(dropped_count, 0);
        }
    }

    #[test]
    fn many_notifications_handled() {
        let mut n: Vec<(&str, u8, u32)> = Vec::new();
        for _ in 0..30 {
            n.push(("msg", 1, 100));
        }
        let v = process(&n, 50, 5);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible.len(), 5);
        }
    }

    #[test]
    fn unicode_message_supported() {
        let n = [("café", 1u8, 100u32)];
        let v = process(&n, 50, 5);
        if let ToastVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["café".to_string()]);
        }
    }
}
