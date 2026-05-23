//! # TUI Sidebar Toggle State Machine
//!
//! Manage sidebar open/closed state across multiple panels with a
//! single "active" sidebar at a time. Toggle returns next-state and
//! whether a redraw is needed.
//!
//! Demonstrates the **TUI.140** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code activitybar single-active-view; tmux pane-toggle
//!  (`prefix + z`) state semantics.
//!
//! Run with: cargo run --example tui_sidebar_toggle_state
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ToggleVerdict {
    Ok {
        active_panel: Option<String>,
        redraw_needed: bool,
    },
    InvalidConfig,
}

pub fn toggle(panels: &[&str], current_active: Option<&str>, requested: &str) -> ToggleVerdict {
    if panels.is_empty() {
        return ToggleVerdict::InvalidConfig;
    }
    if !panels.contains(&requested) {
        return ToggleVerdict::InvalidConfig;
    }
    let new_active = if current_active == Some(requested) {
        None // Toggle off if already active.
    } else {
        Some(requested.to_string())
    };
    let redraw = new_active.as_deref() != current_active;
    ToggleVerdict::Ok {
        active_panel: new_active,
        redraw_needed: redraw,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_sidebar_toggle_state")?;

    let panels = ["explorer", "search", "git"];
    println!("open explorer: {:?}", toggle(&panels, None, "explorer"));
    println!(
        "close explorer: {:?}",
        toggle(&panels, Some("explorer"), "explorer")
    );
    println!("invalid: {:?}", toggle(&panels, None, "unknown"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn open_from_none() {
        let v = toggle(&["a", "b"], None, "a");
        if let ToggleVerdict::Ok { active_panel, .. } = v {
            assert_eq!(active_panel, Some("a".to_string()));
        }
    }

    #[test]
    fn close_when_already_active() {
        let v = toggle(&["a", "b"], Some("a"), "a");
        if let ToggleVerdict::Ok { active_panel, .. } = v {
            assert_eq!(active_panel, None);
        }
    }

    #[test]
    fn switch_between_panels() {
        let v = toggle(&["a", "b"], Some("a"), "b");
        if let ToggleVerdict::Ok { active_panel, .. } = v {
            assert_eq!(active_panel, Some("b".to_string()));
        }
    }

    #[test]
    fn empty_panels_rejected() {
        assert_eq!(toggle(&[], None, "a"), ToggleVerdict::InvalidConfig);
    }

    #[test]
    fn unknown_panel_rejected() {
        let v = toggle(&["a"], None, "unknown");
        assert_eq!(v, ToggleVerdict::InvalidConfig);
    }

    #[test]
    fn redraw_needed_when_open() {
        let v = toggle(&["a", "b"], None, "a");
        if let ToggleVerdict::Ok { redraw_needed, .. } = v {
            assert!(redraw_needed);
        }
    }

    #[test]
    fn redraw_needed_when_close() {
        let v = toggle(&["a", "b"], Some("a"), "a");
        if let ToggleVerdict::Ok { redraw_needed, .. } = v {
            assert!(redraw_needed);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = toggle(&["a", "b"], None, "a");
        let r2 = toggle(&["a", "b"], None, "a");
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_panels_supported() {
        let panels: Vec<&str> = vec!["a", "b", "c", "d", "e", "f"];
        let v = toggle(&panels, Some("c"), "f");
        if let ToggleVerdict::Ok { active_panel, .. } = v {
            assert_eq!(active_panel, Some("f".to_string()));
        }
    }

    #[test]
    fn case_sensitive_panel() {
        let v = toggle(&["Explorer"], None, "explorer");
        assert_eq!(v, ToggleVerdict::InvalidConfig);
    }

    #[test]
    fn unicode_panel_supported() {
        let v = toggle(&["café"], None, "café");
        if let ToggleVerdict::Ok { active_panel, .. } = v {
            assert_eq!(active_panel, Some("café".to_string()));
        }
    }

    #[test]
    fn switch_redraw_needed() {
        let v = toggle(&["a", "b"], Some("a"), "b");
        if let ToggleVerdict::Ok { redraw_needed, .. } = v {
            assert!(redraw_needed);
        }
    }
}
