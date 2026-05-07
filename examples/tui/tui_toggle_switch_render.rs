//! # TUI Toggle Switch Render
//!
//! Render a toggle switch as `[●○]` (off) or `[○●]` (on) with label.
//! Returns rendered string.
//!
//! Demonstrates the **TUI.178** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: iOS UISwitch component; Material Design Switch toggle.
//!
//! Run with: cargo run --example tui_toggle_switch_render
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ToggleVerdict {
    Ok { rendered: String, on: bool },
    InvalidConfig,
}

pub fn render(label: &str, on: bool) -> ToggleVerdict {
    if label.is_empty() {
        return ToggleVerdict::InvalidConfig;
    }
    let switch = if on { "[○●]" } else { "[●○]" };
    ToggleVerdict::Ok {
        rendered: format!("{switch} {label}"),
        on,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_toggle_switch_render")?;

    println!("on: {:?}", render("Notifications", true));
    println!("off: {:?}", render("Notifications", false));
    println!("invalid: {:?}", render("", false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_label_rejected() {
        assert_eq!(render("", true), ToggleVerdict::InvalidConfig);
    }

    #[test]
    fn on_state_renders_correctly() {
        let v = render("X", true);
        if let ToggleVerdict::Ok { rendered, on } = v {
            assert!(rendered.contains("[○●]"));
            assert!(on);
        }
    }

    #[test]
    fn off_state_renders_correctly() {
        let v = render("X", false);
        if let ToggleVerdict::Ok { rendered, on } = v {
            assert!(rendered.contains("[●○]"));
            assert!(!on);
        }
    }

    #[test]
    fn label_in_rendered() {
        let v = render("Notifications", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("Notifications"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("X", true);
        let r2 = render("X", true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn on_state_returned() {
        let v = render("X", true);
        if let ToggleVerdict::Ok { on, .. } = v {
            assert!(on);
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render("café", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn long_label_handled() {
        let v = render("a very long toggle label", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("very long"));
        }
    }

    #[test]
    fn space_separator_present() {
        let v = render("X", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("] "));
        }
    }

    #[test]
    fn switch_glyphs_distinct_per_state() {
        let on = render("X", true);
        let off = render("X", false);
        if let (
            ToggleVerdict::Ok { rendered: r_on, .. },
            ToggleVerdict::Ok {
                rendered: r_off, ..
            },
        ) = (on, off)
        {
            assert_ne!(r_on, r_off);
        }
    }

    #[test]
    fn brackets_present() {
        let v = render("X", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('['));
            assert!(rendered.contains(']'));
        }
    }

    #[test]
    fn rendered_starts_with_switch() {
        let v = render("X", true);
        if let ToggleVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('['));
        }
    }
}
