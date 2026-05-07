//! # TUI Banner Alert Render
//!
//! Render a top-of-screen alert banner with severity-based prefix
//! (INFO/WARN/ERROR/CRITICAL). Returns the rendered banner string.
//!
//! Demonstrates the **TUI.169** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bootstrap alert variants; Material Design snackbar
//!  severity colors.
//!
//! Run with: cargo run --example tui_banner_alert_render
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BannerVerdict {
    Ok { rendered: String, prefix: String },
    InvalidConfig,
}

pub fn render(severity: &str, message: &str, width: u32) -> BannerVerdict {
    if message.is_empty() || width < 20 {
        return BannerVerdict::InvalidConfig;
    }
    let prefix = match severity {
        "info" => "[INFO]",
        "warn" => "[WARN]",
        "error" => "[ERROR]",
        "critical" => "[CRIT]",
        _ => return BannerVerdict::InvalidConfig,
    };
    let inner = format!("{prefix} {message}");
    let truncated = if inner.chars().count() > width as usize {
        let chars: Vec<char> = inner.chars().collect();
        let limit = (width as usize).saturating_sub(1);
        let mut s: String = chars[..limit].iter().collect();
        s.push('…');
        s
    } else {
        inner
    };
    BannerVerdict::Ok {
        rendered: truncated,
        prefix: prefix.to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_banner_alert_render")?;

    println!("info: {:?}", render("info", "Save complete", 40));
    println!("error: {:?}", render("error", "File not found", 40));
    println!("invalid: {:?}", render("xyz", "msg", 40));
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
    fn empty_message_rejected() {
        assert_eq!(render("info", "", 40), BannerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_narrow() {
        assert_eq!(render("info", "hi", 5), BannerVerdict::InvalidConfig);
    }

    #[test]
    fn unknown_severity_rejected() {
        assert_eq!(render("xyz", "hi", 40), BannerVerdict::InvalidConfig);
    }

    #[test]
    fn info_prefix_correct() {
        let v = render("info", "msg", 40);
        if let BannerVerdict::Ok { prefix, .. } = v {
            assert_eq!(prefix, "[INFO]");
        }
    }

    #[test]
    fn warn_prefix_correct() {
        let v = render("warn", "msg", 40);
        if let BannerVerdict::Ok { prefix, .. } = v {
            assert_eq!(prefix, "[WARN]");
        }
    }

    #[test]
    fn error_prefix_correct() {
        let v = render("error", "msg", 40);
        if let BannerVerdict::Ok { prefix, .. } = v {
            assert_eq!(prefix, "[ERROR]");
        }
    }

    #[test]
    fn critical_prefix_correct() {
        let v = render("critical", "msg", 40);
        if let BannerVerdict::Ok { prefix, .. } = v {
            assert_eq!(prefix, "[CRIT]");
        }
    }

    #[test]
    fn message_in_output() {
        let v = render("info", "hello world", 40);
        if let BannerVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("hello world"));
        }
    }

    #[test]
    fn long_message_truncated() {
        let v = render("info", "very long message that exceeds width limit", 25);
        if let BannerVerdict::Ok { rendered, .. } = v {
            assert!(rendered.ends_with('…'));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("info", "msg", 40);
        let r2 = render("info", "msg", 40);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_message_supported() {
        let v = render("info", "café", 40);
        if let BannerVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn min_width_accepted() {
        let v = render("info", "msg", 20);
        assert!(matches!(v, BannerVerdict::Ok { .. }));
    }

    #[test]
    fn case_sensitive_severity() {
        assert_eq!(render("INFO", "msg", 40), BannerVerdict::InvalidConfig);
    }
}
