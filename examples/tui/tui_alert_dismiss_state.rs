//! # TUI Alert Dismissal State
//!
//! Track which alerts have been dismissed (by id) and TTL-expire
//! dismissals after `dismiss_ttl_secs`. Returns the visible alert
//! list at a given time.
//!
//! Demonstrates the **TUI.29** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS notification "snooze" / dismiss patterns.
//!
//! Run with: cargo run --example tui_alert_dismiss_state
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alert {
    pub id: String,
    pub message: String,
    pub created_at_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dismissal {
    pub alert_id: String,
    pub dismissed_at_secs: u64,
}

#[derive(Debug, PartialEq)]
pub enum DismissVerdict {
    Ok {
        visible: Vec<String>,
        dismissed_count: u32,
    },
    InvalidConfig,
}

pub fn visible_at(
    alerts: &[Alert],
    dismissals: &[Dismissal],
    now_secs: u64,
    dismiss_ttl_secs: u64,
) -> DismissVerdict {
    if dismiss_ttl_secs == 0 {
        return DismissVerdict::InvalidConfig;
    }
    let mut active_dismissals: std::collections::BTreeMap<&str, u64> =
        std::collections::BTreeMap::new();
    for d in dismissals {
        let age = now_secs.saturating_sub(d.dismissed_at_secs);
        if age <= dismiss_ttl_secs {
            active_dismissals.insert(d.alert_id.as_str(), d.dismissed_at_secs);
        }
    }
    let mut visible = Vec::new();
    let mut dismissed_count = 0u32;
    for a in alerts {
        if a.created_at_secs > now_secs {
            continue;
        }
        if active_dismissals.contains_key(a.id.as_str()) {
            dismissed_count += 1;
            continue;
        }
        visible.push(a.id.clone());
    }
    DismissVerdict::Ok {
        visible,
        dismissed_count,
    }
}

fn alert(id: &str, msg: &str, created: u64) -> Alert {
    Alert {
        id: id.to_string(),
        message: msg.to_string(),
        created_at_secs: created,
    }
}

fn dismissal(id: &str, at: u64) -> Dismissal {
    Dismissal {
        alert_id: id.to_string(),
        dismissed_at_secs: at,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_alert_dismiss_state")?;

    let alerts = vec![alert("a1", "warn 1", 100), alert("a2", "warn 2", 200)];
    let dismissals = vec![dismissal("a1", 300)];
    println!(
        "fresh dismiss: {:?}",
        visible_at(&alerts, &dismissals, 400, 3600)
    );
    println!(
        "ttl expired: {:?}",
        visible_at(&alerts, &dismissals, 4000, 3600)
    );
    println!("future alert: {:?}", visible_at(&alerts, &[], 50, 3600));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dismisser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fresh_dismissal_hides_alert() {
        let alerts = [alert("a1", "msg", 0)];
        let dismissals = [dismissal("a1", 100)];
        let v = visible_at(&alerts, &dismissals, 200, 3600);
        if let DismissVerdict::Ok {
            visible,
            dismissed_count,
        } = v
        {
            assert!(visible.is_empty());
            assert_eq!(dismissed_count, 1);
        }
    }

    #[test]
    fn ttl_expired_resurfaces() {
        let alerts = [alert("a1", "msg", 0)];
        let dismissals = [dismissal("a1", 100)];
        let v = visible_at(&alerts, &dismissals, 10_000, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["a1".to_string()]);
        }
    }

    #[test]
    fn future_alert_excluded() {
        let alerts = [alert("a1", "msg", 1000)];
        let v = visible_at(&alerts, &[], 100, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert!(visible.is_empty());
        }
    }

    #[test]
    fn invalid_zero_ttl() {
        assert_eq!(visible_at(&[], &[], 100, 0), DismissVerdict::InvalidConfig);
    }

    #[test]
    fn no_dismissals_all_visible() {
        let alerts = vec![alert("a1", "x", 0), alert("a2", "y", 0)];
        let v = visible_at(&alerts, &[], 100, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert_eq!(visible.len(), 2);
        }
    }

    #[test]
    fn unknown_dismissal_ignored() {
        let alerts = [alert("a1", "msg", 0)];
        let dismissals = [dismissal("ghost", 100)];
        let v = visible_at(&alerts, &dismissals, 200, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["a1".to_string()]);
        }
    }

    #[test]
    fn boundary_at_ttl_dismissed() {
        // Age == ttl → still dismissed.
        let alerts = [alert("a1", "msg", 0)];
        let dismissals = [dismissal("a1", 100)];
        let v = visible_at(&alerts, &dismissals, 100 + 3600, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert!(visible.is_empty());
        }
    }

    #[test]
    fn multiple_alerts_partial() {
        let alerts = vec![alert("a1", "x", 0), alert("a2", "y", 0)];
        let dismissals = vec![dismissal("a1", 100)];
        let v = visible_at(&alerts, &dismissals, 200, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["a2".to_string()]);
        }
    }

    #[test]
    fn no_alerts_empty_visible() {
        let v = visible_at(&[], &[], 100, 3600);
        if let DismissVerdict::Ok { visible, .. } = v {
            assert!(visible.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let alerts = vec![alert("a1", "x", 0)];
        let dismissals = vec![dismissal("a1", 100)];
        let a = visible_at(&alerts, &dismissals, 200, 3600);
        let b = visible_at(&alerts, &dismissals, 200, 3600);
        assert_eq!(a, b);
    }
}
