//! # TUI Toast Notification Render
//!
//! Render a toast notification line: `[ ✓ Saved          ]` with
//! optional timer countdown. Returns rendered string and whether the
//! toast has expired.
//!
//! Demonstrates the **TUI.100** recipe for PMAT-193 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS NSUserNotification dismiss timer; Material Design
//!  Snackbar duration spec.
//!
//! Run with: cargo run --example tui_toast_notification_render
//!
//! Added by PMAT-193 (catalog 1360→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ToastKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, PartialEq)]
pub enum ToastVerdict {
    Ok { rendered: String, expired: bool },
    InvalidConfig,
}

pub fn render(
    kind: ToastKind,
    message: &str,
    width: u32,
    elapsed_ms: u32,
    timeout_ms: u32,
) -> ToastVerdict {
    if message.is_empty() || width < 6 || timeout_ms == 0 {
        return ToastVerdict::InvalidConfig;
    }
    let icon = match kind {
        ToastKind::Info => 'i',
        ToastKind::Success => '+',
        ToastKind::Warning => '!',
        ToastKind::Error => 'x',
    };
    // Layout: "[ X message ]" with padding to width.
    let inner_max = width as usize - 4; // brackets + spaces
    let truncated_msg: String = message.chars().take(inner_max - 2).collect();
    let display = format!("{icon} {truncated_msg}");
    let pad = inner_max - display.chars().count();
    let rendered = format!("[ {display}{} ]", " ".repeat(pad));
    let expired = elapsed_ms >= timeout_ms;
    ToastVerdict::Ok { rendered, expired }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_toast_notification_render")?;

    println!(
        "info: {:?}",
        render(ToastKind::Info, "Loaded", 30, 1000, 5000)
    );
    println!(
        "expired: {:?}",
        render(ToastKind::Success, "Saved", 30, 6000, 5000)
    );
    println!("invalid: {:?}", render(ToastKind::Info, "", 30, 0, 5000));
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
    fn info_renders_brackets() {
        let v = render(ToastKind::Info, "Hi", 20, 0, 5000);
        if let ToastVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('['));
            assert!(rendered.ends_with(']'));
        }
    }

    #[test]
    fn icon_per_kind() {
        let info = render(ToastKind::Info, "x", 20, 0, 5000);
        let success = render(ToastKind::Success, "x", 20, 0, 5000);
        if let (ToastVerdict::Ok { rendered: i, .. }, ToastVerdict::Ok { rendered: s, .. }) =
            (info, success)
        {
            assert!(i.contains('i'));
            assert!(s.contains('+'));
        }
    }

    #[test]
    fn expired_after_timeout() {
        let v = render(ToastKind::Info, "Hi", 20, 6000, 5000);
        if let ToastVerdict::Ok { expired, .. } = v {
            assert!(expired);
        }
    }

    #[test]
    fn not_expired_within_timeout() {
        let v = render(ToastKind::Info, "Hi", 20, 1000, 5000);
        if let ToastVerdict::Ok { expired, .. } = v {
            assert!(!expired);
        }
    }

    #[test]
    fn empty_message_rejected() {
        assert_eq!(
            render(ToastKind::Info, "", 20, 0, 5000),
            ToastVerdict::InvalidConfig
        );
    }

    #[test]
    fn too_small_width_rejected() {
        assert_eq!(
            render(ToastKind::Info, "Hi", 5, 0, 5000),
            ToastVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_timeout_rejected() {
        assert_eq!(
            render(ToastKind::Info, "Hi", 20, 0, 0),
            ToastVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let r1 = render(ToastKind::Info, "Hi", 20, 0, 5000);
        let r2 = render(ToastKind::Info, "Hi", 20, 0, 5000);
        assert_eq!(r1, r2);
    }

    #[test]
    fn message_truncated_to_width() {
        let v = render(ToastKind::Info, "very long toast message", 15, 0, 5000);
        if let ToastVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.chars().count(), 15);
        }
    }

    #[test]
    fn rendered_at_least_brackets_and_icon() {
        let v = render(ToastKind::Warning, "Hi", 10, 0, 5000);
        if let ToastVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('!'));
            assert!(rendered.contains("Hi"));
        }
    }

    #[test]
    fn boundary_at_timeout_expired() {
        let v = render(ToastKind::Info, "Hi", 20, 5000, 5000);
        if let ToastVerdict::Ok { expired, .. } = v {
            assert!(expired);
        }
    }
}
