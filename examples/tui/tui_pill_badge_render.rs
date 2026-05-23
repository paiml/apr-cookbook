//! # TUI Pill Badge Render
//!
//! Render a pill-shaped badge with label and optional count.
//! Returns the rendered string with rounded-corner glyphs.
//!
//! Demonstrates the **TUI.171** recipe for PMAT-219 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Material Design chip/badge component; iOS notification
//!  badge convention.
//!
//! Run with: cargo run --example tui_pill_badge_render
//!
//! Added by PMAT-219 (catalog 1594→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PillVerdict {
    Ok { rendered: String, width: u32 },
    InvalidConfig,
}

pub fn render(label: &str, count: Option<u32>) -> PillVerdict {
    if label.is_empty() {
        return PillVerdict::InvalidConfig;
    }
    let inner = match count {
        Some(c) if c > 99 => format!("{label} 99+"),
        Some(c) => format!("{label} {c}"),
        None => label.to_string(),
    };
    let rendered = format!("({inner})");
    let width = rendered.chars().count() as u32;
    PillVerdict::Ok { rendered, width }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_pill_badge_render")?;

    println!("label: {:?}", render("New", None));
    println!("count: {:?}", render("Inbox", Some(5)));
    println!("99+: {:?}", render("Mentions", Some(150)));
    println!("invalid: {:?}", render("", None));
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
        assert_eq!(render("", None), PillVerdict::InvalidConfig);
    }

    #[test]
    fn label_only_rendered() {
        let v = render("New", None);
        if let PillVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "(New)");
        }
    }

    #[test]
    fn count_appended() {
        let v = render("Inbox", Some(5));
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("Inbox 5"));
        }
    }

    #[test]
    fn count_over_99_truncated() {
        let v = render("X", Some(150));
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("99+"));
        }
    }

    #[test]
    fn parens_present() {
        let v = render("X", None);
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with('('));
            assert!(rendered.ends_with(')'));
        }
    }

    #[test]
    fn width_correct() {
        let v = render("ab", None);
        if let PillVerdict::Ok { width, .. } = v {
            // "(ab)" → 4 chars
            assert_eq!(width, 4);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("X", None);
        let r2 = render("X", None);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_zero_included() {
        let v = render("X", Some(0));
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("0"));
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render("café", None);
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
        }
    }

    #[test]
    fn count_99_no_plus() {
        let v = render("X", Some(99));
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(!rendered.contains("+"));
        }
    }

    #[test]
    fn count_100_shows_plus() {
        let v = render("X", Some(100));
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("99+"));
        }
    }

    #[test]
    fn long_label_handled() {
        let v = render("a very long badge label", None);
        if let PillVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("very long"));
        }
    }
}
